import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score
import main                      # CNN + DINO head (head.pt)        -> fracture yes/no
import FractureLocator as fl     # patch head (fracture_head.pt)    -> fracture yes/no + box

# Both trained heads vote together, using Monte Carlo dropout.
#
# Normally dropout is OFF when predicting. For Monte Carlo dropout we leave it ON and predict
# PASSES times per head: every pass drops different neurons -> a slightly different yes/no.
#   5 passes CNN+DINO head + 5 passes FractureLocator head = 10 votes per image
#   final answer = majority vote: "fracture" if at least half say fracture
#                  (a 5-5 tie counts as fracture: missing a fracture is worse than a false alarm)
#   confidence   = share of the 10 votes that agree with the final answer
# The box comes from FractureLocator's average heatmap over its passes.
#
# Run after main.py and FractureLocator.py.

PASSES = 5
SEED = 42
torch.manual_seed(SEED)
device = main.device
fracture = main.label_map["fracture"]           # which output column of the CNN+DINO head is "fracture"

# the two trained heads
val_cnn, val_dino, val_targets = main.load("val")
head = main.Head(val_cnn.shape[1], val_dino.shape[1], main.num_classes).to(device)
head.load_state_dict(torch.load("head.pt"))

val_data, val_patches, val_masks, val_labels = fl.load("val", keep_flipped=False)
locator = fl.PatchHead(val_patches.shape[-1]).to(device)
locator.load_state_dict(torch.load("fracture_head.pt"))

# one threshold per head, from VAL with dropout off: the highest that still catches 90% of the fractures
head.eval()
with torch.no_grad():
    head_threshold = main.pick_thresholds(torch.sigmoid(head(val_cnn, val_dino)), val_targets)[fracture]
val_square_scores = fl.predict_all(locator, val_patches)
locator_threshold = fl.pick_threshold(torch.sigmoid(fl.image_scores(val_square_scores)), val_labels)
print("thresholds: CNN+DINO", round(head_threshold, 3), " FractureLocator", round(locator_threshold, 3))

all_scores = []
for split in ["val", "test"]:
    cnn, dino, targets = main.load(split)
    data, patches, masks, labels = fl.load(split, keep_flipped=False)
    image_paths = pd.read_csv(f"{main.DATA}/{split}.csv")["image_path"]
    assert list(image_paths) == list(data["image_path"]), "both heads must see the images in the same order"
    y = targets[:, fracture].cpu().numpy()      # truth: 1 = fracture, 0 = normal

    votes = []                                  # one row of N yes/no votes per pass
    heatmaps = []                               # FractureLocator square probabilities per pass, for the box

    head.train()                                # dropout ON = Monte Carlo dropout
    with torch.no_grad():
        for _ in range(PASSES):
            probs = torch.sigmoid(head(cnn, dino))[:, fracture].cpu()
            votes.append((probs >= head_threshold).float())

    for _ in range(PASSES):
        square_scores = fl.predict_all(locator, patches, mc_dropout=True)   # [N, 32, 32], dropout ON
        probs = torch.sigmoid(fl.image_scores(square_scores))
        votes.append((probs >= locator_threshold).float())
        heatmaps.append(torch.sigmoid(square_scores))

    votes = torch.stack(votes)                  # [10, N]
    fracture_share = votes.mean(0).numpy()      # share of the 10 votes that say fracture, 0-1
    predicted = (fracture_share >= 0.5).astype(int)
    confidence = np.maximum(fracture_share, 1 - fracture_share)
    mean_heatmap = torch.stack(heatmaps).mean(0).numpy()   # [N, 32, 32]

    # one row per image
    rows = []
    for i in range(len(y)):
        box = ""
        if predicted[i] == 1:                   # box only when the final answer is "fracture"
            x1, y1, x2, y2 = fl.find_box(mean_heatmap[i])
            box = [round(x1), round(y1), round(x2), round(y2)]
        if predicted[i] == 1:
            answer = "fracture"
        else:
            answer = "normal"
        rows.append([image_paths[i], data["view"][i], data["label"][i], int(votes[:, i].sum()),
                     answer, round(float(confidence[i]), 2), box])
    results = pd.DataFrame(rows, columns=["image_path", "view", "true", "fracture_votes_of_10",
                                          "predicted", "confidence", "box"])
    results.to_csv(f"vote_{split}_predictions.csv", index=False)

    # pictures with only the box (no heatmap): results/boxes_{split}/
    os.makedirs(f"results/boxes_{split}", exist_ok=True)
    for i in range(len(results)):
        img = Image.open(image_paths[i]).convert("RGB")      # 1024 x 1024, same pixels as the box
        if results["box"][i] != "":
            ImageDraw.Draw(img).rectangle(results["box"][i], outline=(255, 255, 0), width=4)
        percent = round(results["confidence"][i] * 100)
        name = f"{results['true'][i]}_said_{results['predicted'][i]}_conf{percent}_{os.path.basename(image_paths[i])}"
        img.save(f"results/boxes_{split}/{name}")

    # final answer vs the truth
    right = predicted == y
    scores = {
        "split": split,
        "f1": round(f1_score(y, predicted), 3),
        "accuracy": round(accuracy_score(y, predicted), 3),
        "recall": round(recall_score(y, predicted), 3),          # fractures caught
        "precision": round(precision_score(y, predicted), 3),    # "fracture" answers that were right
        "auroc": round(roc_auc_score(y, fracture_share), 3),
        "confidence_when_right": round(float(confidence[right].mean()), 3),
        "confidence_when_wrong": None,
        "mistakes": int((~right).sum()),
    }
    if (~right).any():
        scores["confidence_when_wrong"] = round(float(confidence[~right].mean()), 3)
    all_scores.append(scores)

    # ROC curve: for every vote cut-off (1 of 10, 2 of 10, ...) how many fractures are caught
    # (true positive rate) vs how many normals get a false alarm (false positive rate)
    fpr, tpr, _ = roc_curve(y, fracture_share)
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, marker="o", label=f"majority vote (AUROC {scores['auroc']})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="coin flip")
    plt.xlabel("false positive rate (normals called fracture)")
    plt.ylabel("true positive rate (fractures caught)")
    plt.title(f"ROC curve - {split}")
    plt.legend()
    plt.savefig(f"roc_{split}.png", dpi=120)
    plt.close()

all_scores = pd.DataFrame(all_scores)
all_scores.to_csv("vote_metrics.csv", index=False)
print()
print("majority vote of both heads:")
print(all_scores.to_string(index=False))
