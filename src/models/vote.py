import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score
# ---------- always work from the project folder (X-ray-ml) ----------
# This file lives in src/<folder>/, but all paths below ("data/...", "src/models/weights/...") start at the
# project folder. These lines go there first, so it works from PyCharm's Run button AND from a terminal.
PROJECT_FOLDER = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # up 3: file -> src/<folder> -> src -> project
os.chdir(PROJECT_FOLDER)

import main                      # CNN + DINO head (head.pt)        -> fracture yes/no
import FractureLocator as fl     # patch head (fracture_head.pt)    -> fracture yes/no + box

# ======================================================================================
# vote.py - both trained heads vote together, using Monte Carlo dropout
#
# Normally dropout is OFF when predicting. For Monte Carlo dropout we leave it ON and predict
# PASSES times per head: every pass switches off different neurons -> a slightly different yes/no.
#   5 passes of the CNN+DINO head + 5 passes of the FractureLocator head = 10 votes per X-ray
#   final answer = majority vote: "fracture" if at least 5 of the 10 votes say fracture
#                  (a 5-5 tie counts as fracture: missing a fracture is worse than a false alarm)
#   confidence   = share of the 10 votes that agree with the final answer
# The box comes from FractureLocator's average heatmap over its 5 passes.
#
# Run after main.py and FractureLocator.py (src/models/).
# ======================================================================================

PASSES = 5
SEED = 42
torch.manual_seed(SEED)
device = main.device
fracture = main.label_map["fracture"]           # which output of the CNN+DINO head means "fracture"


# ---------- the two trained heads ----------
val_cnn, val_dino, val_targets = main.load("val")
head = main.Head(val_cnn.shape[1], val_dino.shape[1], main.num_classes).to(device)
head.load_state_dict(torch.load("src/models/weights/head.pt"))

val_data, val_patches, val_masks, val_labels = fl.load("val", keep_flipped=False)
locator = fl.PatchHead(val_patches.shape[-1]).to(device)
locator.load_state_dict(torch.load("src/models/weights/fracture_head.pt"))


# ---------- one threshold per head ----------
# chosen on VAL with dropout off: the highest threshold that still catches 90% of the val fractures
head.eval()
with torch.no_grad():
    val_probs = torch.sigmoid(head(val_cnn, val_dino))
head_threshold = main.pick_thresholds(val_probs, val_targets)[fracture]

val_square_scores = fl.predict_all(locator, val_patches)
locator_threshold = fl.pick_threshold(torch.sigmoid(fl.image_scores(val_square_scores)), val_labels)
print("thresholds: CNN+DINO", round(head_threshold, 3), " FractureLocator", round(locator_threshold, 3))


all_scores = []
for split in ["val", "test"]:
    cnn, dino, targets = main.load(split)
    data, patches, masks, labels = fl.load(split, keep_flipped=False)
    image_paths = pd.read_csv(f"{main.DATA}/{split}.csv")["image_path"]
    # both heads must see the X-rays in the same order
    if list(image_paths) != list(data["image_path"]):
        raise ValueError("main.py and FractureLocator rows are in a different order")
    truth = targets[:, fracture].cpu().numpy()      # 1 = fracture, 0 = normal
    count = len(truth)

    # ---------- the 10 passes ----------
    pass_probs = []          # 5 lists of CNN+DINO probabilities, then 5 lists of FractureLocator probabilities
    pass_heatmaps = []       # 5 FractureLocator heatmaps per X-ray (for the box)

    head.train()                                    # dropout ON = Monte Carlo dropout
    with torch.no_grad():
        for p in range(PASSES):
            probs = torch.sigmoid(head(cnn, dino))[:, fracture].cpu().numpy()
            pass_probs.append(["CNN+DINO", probs])

    for p in range(PASSES):
        square_scores = fl.predict_all(locator, patches, mc_dropout=True)    # [N, 32, 32], dropout ON
        probs = torch.sigmoid(fl.image_scores(square_scores)).numpy()
        pass_probs.append(["FractureLocator", probs])
        pass_heatmaps.append(torch.sigmoid(square_scores).numpy())

    # ---------- votes, answer and confidence for every X-ray ----------
    fracture_votes = []      # how many of the 10 votes say "fracture"
    predicted = []           # 1 = fracture, 0 = normal
    confidence = []          # share of votes agreeing with the answer
    rows = []
    for i in range(count):
        votes = 0
        for head_name, probs in pass_probs:
            if head_name == "CNN+DINO":
                threshold = head_threshold
            else:
                threshold = locator_threshold
            if probs[i] >= threshold:
                votes = votes + 1
        share = votes / len(pass_probs)             # e.g. 8 of 10 -> 0.8

        if share >= 0.5:
            answer = 1
            agree = share
        else:
            answer = 0
            agree = 1 - share
        fracture_votes.append(votes)
        predicted.append(answer)
        confidence.append(agree)

        # the box: average the 5 heatmaps of this X-ray, then FractureLocator's box rule
        box = ""
        if answer == 1:                              # box only when the answer is "fracture"
            average_heatmap = np.zeros((32, 32))
            for heatmaps in pass_heatmaps:
                average_heatmap = average_heatmap + heatmaps[i] / PASSES
            x1, y1, x2, y2 = fl.find_box(average_heatmap)
            box = [round(x1), round(y1), round(x2), round(y2)]

        if answer == 1:
            answer_text = "fracture"
        else:
            answer_text = "normal"
        rows.append([image_paths[i], data["view"][i], data["label"][i], votes, answer_text, round(agree, 2), box])

    results = pd.DataFrame(rows, columns=["image_path", "view", "true", "fracture_votes_of_10",
                                          "predicted", "confidence", "box"])
    results.to_csv(f"metrics/vote_{split}_predictions.csv", index=False)

    # ---------- pictures with only the box (no heatmap): outputs/boxes_{split}/ ----------
    os.makedirs(f"metrics/boxes_{split}", exist_ok=True)
    for i in range(len(results)):
        img = Image.open(image_paths[i]).convert("RGB")      # 1024 x 1024, same pixels as the box
        if results["box"][i] != "":
            ImageDraw.Draw(img).rectangle(results["box"][i], outline=(255, 255, 0), width=4)
        percent = round(results["confidence"][i] * 100)
        name = f"{results['true'][i]}_said_{results['predicted'][i]}_conf{percent}_{os.path.basename(image_paths[i])}"
        img.save(f"metrics/boxes_{split}/{name}")

    # ---------- final answer vs the truth ----------
    fracture_share = []
    for votes in fracture_votes:
        fracture_share.append(votes / len(pass_probs))

    right_confidence = []
    wrong_confidence = []
    for i in range(count):
        if predicted[i] == truth[i]:
            right_confidence.append(confidence[i])
        else:
            wrong_confidence.append(confidence[i])

    scores = {
        "split": split,
        "f1": round(f1_score(truth, predicted), 3),
        "accuracy": round(accuracy_score(truth, predicted), 3),
        "recall": round(recall_score(truth, predicted), 3),          # fractures caught
        "precision": round(precision_score(truth, predicted), 3),    # "fracture" answers that were right
        "auroc": round(roc_auc_score(truth, fracture_share), 3),
        "confidence_when_right": round(float(np.mean(right_confidence)), 3),
        "confidence_when_wrong": None,
        "mistakes": len(wrong_confidence),
    }
    if len(wrong_confidence) > 0:
        scores["confidence_when_wrong"] = round(float(np.mean(wrong_confidence)), 3)
    all_scores.append(scores)

    # ---------- ROC curve ----------
    # for every vote cut-off (1 of 10, 2 of 10, ...): how many fractures are caught (true positive rate)
    # vs how many normals get a false alarm (false positive rate)
    fpr, tpr, cutoffs = roc_curve(truth, fracture_share)
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, marker="o", label=f"majority vote (AUROC {scores['auroc']})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="coin flip")
    plt.xlabel("false positive rate (normals called fracture)")
    plt.ylabel("true positive rate (fractures caught)")
    plt.title(f"ROC curve - {split}")
    plt.legend()
    plt.savefig(f"metrics/roc_{split}.png", dpi=120)
    plt.close()

all_scores = pd.DataFrame(all_scores)
all_scores.to_csv("metrics/vote_metrics.csv", index=False)
print()
print("majority vote of both heads:")
print(all_scores.to_string(index=False))
