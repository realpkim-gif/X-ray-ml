import os
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.optim as optim
from PIL import Image, ImageDraw
from sklearn.metrics import roc_auc_score
from main import make_scheduler, device, TARGET_RECALL   # same lr schedule and recall goal as main.py

# Finds fractures AND roughly where they are (a box), from the DINO patch vectors made by DINO_patches.py.
# The head scores every square of the 32 x 32 grid: "is this square part of a fracture?"
#   patches [B, 32, 32, 768] -> head -> square scores [B, 32, 32] -> heatmap + box (where)
#   image score = average of the TOP_K highest squares             -> fracture yes/no
# Squares outside the outline in fracture images, and all squares of normal images, are "healthy".
# So the head has to learn what a fracture looks like, not which kind of X-ray it is.

FEATURES = "data/features"
VIEWS = "data/views.csv"    # body / limb / other_source per image (from clustering, checked by eye)
EPOCHS = 40
BATCH_SIZE = 16
LR = 0.001
TOP_K = 8                   # image score = mean of the 8 highest squares (1 = only the highest square)
IMAGE_WEIGHT = 1.0          # how much the per-image loss (fracture yes/no) counts
PATCH_WEIGHT = 1.0          # how much the per-square loss (inside the outline or not) counts
PATCH_POS_WEIGHT = 30.0     # fracture squares are rare (~3%), so each one counts 30x
BALANCE_VIEWS = True        # weight images so fracture/normal are 50/50 inside every view (no "view" shortcut)
BOX_THRESHOLD = 0.3         # the box grows from the hottest square into touching squares at least this share as hot
SEED = 42
torch.manual_seed(SEED)


def load(split, keep_flipped):
    # keep_flipped: mirrored copies are extra TRAINING examples; val/test are scored on the real images only.
    # everything stays on the cpu; each batch is moved to the gpu when it is used (saves gpu memory).
    data = pd.read_csv(f"{FEATURES}/patches_{split}.csv")
    views = pd.read_csv(VIEWS)
    data = data.merge(views, on="original", how="left")      # adds the "view" column
    patches = np.load(f"{FEATURES}/patches_{split}.npy")     # [N, 32, 32, 768] half precision
    masks = np.load(f"{FEATURES}/masks_{split}.npy")         # [N, 32, 32] 0-1, share of each square inside the outline

    if not keep_flipped:
        keep = (data["flipped"] == False).values
        patches = patches[keep]
        masks = masks[keep]
        data = data[keep].reset_index(drop=True)

    labels = []                                              # 1 = fracture, 0 = normal
    for label in data["label"]:
        if label == "fracture":
            labels.append(1.0)
        else:
            labels.append(0.0)
    return data, torch.tensor(patches), torch.tensor(masks), torch.tensor(labels)


class PatchHead(nn.Module):
    def __init__(self, dino_size):
        super(PatchHead, self).__init__()
        self.norm = nn.LayerNorm(dino_size)              # every square vector on the same scale
        self.conv1 = nn.Conv2d(dino_size, 256, 1)        # 1x1: each square on its own   [B, 768, 32, 32] -> [B, 256, 32, 32]
        self.conv2 = nn.Conv2d(256, 256, 3, padding=1)   # 3x3: each square also looks at its 8 neighbours
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(0.2)                 # randomly drop whole channels while training (less overfitting)
        self.conv3 = nn.Conv2d(256, 1, 1)                # one raw score per square        [B, 1, 32, 32]

    def forward(self, patches):
        x = self.norm(patches.float())                   # [B, 32, 32, 768]
        x = x.permute(0, 3, 1, 2)                        # conv layers want channels first: [B, 768, 32, 32]
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.dropout(x)
        x = self.conv3(x)                                # [B, 1, 32, 32]
        return x[:, 0]                                   # [B, 32, 32] raw square scores


def image_scores(square_scores):
    # [B, 32, 32] -> [B]: one score per image = mean of its TOP_K highest squares. One clear fracture spot is enough.
    flat = square_scores.reshape(len(square_scores), -1)   # all 1024 squares of an image in one row [B, 1024]
    highest = torch.topk(flat, TOP_K, dim=1).values          # the TOP_K highest of each row        [B, TOP_K]
    return highest.mean(dim=1)                               # [B]


def view_weights(data):
    # each (view, label) group gets the same total weight,
    # e.g. 65 body fractures count as much as 443 body normals together
    weights = np.ones(len(data))
    if BALANCE_VIEWS:
        groups = (data["view"].astype(str) + "_" + data["label"]).values   # e.g. "body_normal"
        counts = {}
        for group in groups:                                 # how many images in every group
            counts[group] = counts.get(group, 0) + 1
        for i in range(len(data)):
            weights[i] = 1 / counts[groups[i]]
        weights = weights / weights.mean()                   # average weight stays 1
    return torch.tensor(weights, dtype=torch.float32)


def predict_all(model, patches):
    # square scores for every image, 64 images at a time -> [N, 32, 32] (on the cpu)
    model.eval()                                           # dropout off
    scores = []
    with torch.no_grad():
        for start in range(0, len(patches), 64):
            batch = patches[start:start + 64].to(device)
            scores.append(model(batch).cpu())
    return torch.cat(scores)


def view_aurocs(data, labels, probs):
    # AUROC on everything, and separately inside each view (the honest numbers: no view shortcut possible)
    y = labels.numpy()
    p = probs.numpy()
    result = {"all": roc_auc_score(y, p)}
    for view in ["limb", "body"]:
        inside = (data["view"] == view).values
        if 0 < y[inside].sum() < inside.sum():               # needs both fractures and normals
            result[view] = roc_auc_score(y[inside], p[inside])
    per_view = []
    for view in ["limb", "body"]:
        if view in result:
            per_view.append(result[view])
    result["view_avg"] = np.mean(per_view)
    return result


def find_box(square_probs):
    # 1. start at the hottest square
    # 2. keep adding touching squares (up/down/left/right) that are at least BOX_THRESHOLD x as hot as it
    # 3. one box around all added squares, in pixels of the 1024 x 1024 image
    grid = square_probs.shape[0]
    limit = BOX_THRESHOLD * square_probs.max()
    start = np.unravel_index(square_probs.argmax(), square_probs.shape)   # (row, col) of the hottest square
    blob = [start]          # squares in the box
    todo = [start]          # squares whose neighbours we still have to check
    while len(todo) > 0:
        row, col = todo.pop()
        for r, c in [(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)]:
            inside_grid = 0 <= r < grid and 0 <= c < grid
            if inside_grid and (r, c) not in blob and square_probs[r, c] >= limit:
                blob.append((r, c))
                todo.append((r, c))

    rows = [r for r, c in blob]
    cols = [c for r, c in blob]
    scale = 1024 / grid     # one square = 32 pixels
    return [min(cols) * scale, min(rows) * scale, (max(cols) + 1) * scale, (max(rows) + 1) * scale]


def box_from_mask(mask):
    # the real fracture box: around every square that touches the outline
    rows, cols = np.where(mask > 0)
    scale = 1024 / mask.shape[0]
    return [cols.min() * scale, rows.min() * scale, (cols.max() + 1) * scale, (rows.max() + 1) * scale]


def box_iou(a, b):
    # overlap of two boxes [x1, y1, x2, y2]: 0 = no overlap, 1 = identical
    width = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    overlap = width * height
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return overlap / (area_a + area_b - overlap)


def pick_threshold(probs, labels):
    # the highest threshold that still catches TARGET_RECALL (90%) of the fractures
    fracture_probs = np.sort(probs[labels == 1].numpy())     # low -> high
    allowed_misses = int(len(fracture_probs) * (1 - TARGET_RECALL))
    return float(fracture_probs[allowed_misses])


def evaluate(data, masks, labels, square_scores, threshold):
    # one row per image (prediction + box) and one dict of scores
    square_probs = torch.sigmoid(square_scores).numpy()     # [N, 32, 32] 0-1
    probs = torch.sigmoid(image_scores(square_scores))      # [N] 0-1
    aurocs = view_aurocs(data, labels, probs)
    y = labels.numpy()
    p = probs.numpy()
    predicted = (p >= threshold).astype(int)

    rows = []
    hits = []               # fracture images: is the hottest square inside the real outline?
    ious = []               # fracture images: overlap of our box with the real box
    for i in range(len(data)):
        box = find_box(square_probs[i])
        hit = ""
        iou = ""
        if y[i] == 1:
            mask = masks[i].numpy()
            top = np.unravel_index(square_probs[i].argmax(), mask.shape)
            hit = bool(mask[top] > 0)
            iou = round(box_iou(box, box_from_mask(mask)), 3)
            hits.append(hit)
            ious.append(iou)
        if predicted[i] == 1:
            prediction = "fracture"
        else:
            prediction = "normal"
        rows.append([data["image_path"][i], data["view"][i], data["label"][i], round(float(p[i]), 3), prediction,
                     round(box[0]), round(box[1]), round(box[2]), round(box[3]), hit, iou])
    results = pd.DataFrame(rows, columns=["image_path", "view", "true", "probability", "predicted",
                                         "box_x1", "box_y1", "box_x2", "box_y2", "hottest_square_in_outline", "box_iou"])

    scores = {
        "auroc_all": round(aurocs["all"], 3),
        "auroc_limb": round(aurocs.get("limb", float("nan")), 3),
        "auroc_body": round(aurocs.get("body", float("nan")), 3),
        "threshold": round(threshold, 3),
        "recall": float(round(predicted[y == 1].mean(), 3)),          # fractures caught
        "specificity": float(round(1 - predicted[y == 0].mean(), 3)), # normals correctly left alone
        "accuracy": float(round((predicted == y).mean(), 3)),
        "location_hit_rate": float(round(np.mean(hits), 3)),          # hottest square inside the real outline
        "box_iou_mean": float(round(np.mean(ious), 3)),               # how well our box overlaps the real one
        "fractures": int(y.sum()),
        "normals": int(len(y) - y.sum()),
    }
    return results, scores


def save_heatmap(path, square_probs, box, out_path):
    # X-ray with the fracture probability painted red on top + the predicted box in yellow
    img = Image.open(path).convert("RGB").resize((512, 512))
    heat = Image.fromarray((square_probs * 255).astype(np.uint8)).resize((512, 512), Image.BILINEAR)
    heat = heat.point(lambda v: int(v * 0.6))                # at most 60% red, so the bone stays visible
    red = Image.new("RGB", img.size, (255, 0, 0))
    img = Image.composite(red, img, heat)
    if box is not None:                                      # only drawn when the model says "fracture"
        box = [box[0] / 2, box[1] / 2, box[2] / 2, box[3] / 2]   # 1024 pixels -> 512 pixels
        ImageDraw.Draw(img).rectangle(box, outline=(255, 255, 0), width=2)
    img.save(out_path)


def train_head(train_data, train_patches, train_masks, train_labels, val=None, save_path=None):
    # trains one PatchHead. val = (data, patches, masks, labels): then the epoch with the best
    # view-averaged val AUROC is kept (saved to save_path). Without val the last epoch is kept.
    train_weights = view_weights(train_data)
    model = PatchHead(train_patches.shape[-1]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)  # weight_decay = less overfitting
    steps_per_epoch = (len(train_patches) + BATCH_SIZE - 1) // BATCH_SIZE
    scheduler = make_scheduler(optimizer, EPOCHS * steps_per_epoch)
    square_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(PATCH_POS_WEIGHT, device=device))
    image_loss = nn.BCEWithLogitsLoss(reduction="none")   # "none": one loss per image, so we can weight them
    best_val = -1
    history = []

    for epoch in range(EPOCHS):
        # training
        model.train()                               # dropout on
        batch_losses = []
        order = torch.randperm(len(train_patches))  # shuffle
        for start in range(0, len(order), BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            patches = train_patches[idx].to(device)   # [B, 32, 32, 768]
            masks = train_masks[idx].to(device)       # [B, 32, 32]
            labels = train_labels[idx].to(device)     # [B]
            weights = train_weights[idx].to(device)   # [B]

            optimizer.zero_grad()
            scores = model(patches)                   # [B, 32, 32]
            loss_images = image_loss(image_scores(scores), labels)   # fracture yes/no, one loss per image [B]
            loss_images = (loss_images * weights).mean()
            loss_squares = square_loss(scores, masks)                # every square: inside the outline or not
            loss = IMAGE_WEIGHT * loss_images + PATCH_WEIGHT * loss_squares
            loss.backward()
            optimizer.step()
            scheduler.step()
            batch_losses.append(loss.item())
        train_loss = sum(batch_losses) / len(batch_losses)
        if val is None:
            continue

        # validation: keep the epoch with the best view-averaged AUROC (can't be won by guessing the view)
        val_data, val_patches, val_masks, val_labels = val
        val_scores = predict_all(model, val_patches)
        aurocs = view_aurocs(val_data, val_labels, torch.sigmoid(image_scores(val_scores)))
        history.append([epoch + 1, train_loss, aurocs["all"], aurocs["view_avg"]])
        if aurocs["view_avg"] > best_val:
            best_val = aurocs["view_avg"]
            torch.save(model.state_dict(), save_path)
        print(f"Epoch [{epoch + 1}/{EPOCHS}], Train loss: {train_loss:.4f}, "
              f"val AUROC all {aurocs['all']:.3f} limb {aurocs.get('limb', 0):.3f} body {aurocs.get('body', 0):.3f}")

    if val is not None:
        print("best val view-averaged AUROC:", round(best_val, 3), "saved to", save_path)
        model.load_state_dict(torch.load(save_path))
    return model, history


# everything below only runs when you run this file itself (cross_validate.py imports the functions above)
if __name__ == "__main__":
    train_data, train_patches, train_masks, train_labels = load("train", keep_flipped=True)
    val_data, val_patches, val_masks, val_labels = load("val", keep_flipped=False)
    test_data, test_patches, test_masks, test_labels = load("test", keep_flipped=False)

    val = (val_data, val_patches, val_masks, val_labels)
    model, history = train_head(train_data, train_patches, train_masks, train_labels, val, "fracture_head.pt")
    history = pd.DataFrame(history, columns=["epoch", "train_loss", "val_auroc", "val_auroc_view_avg"])
    history.to_csv("fracture_trainHistory.csv", index=False)


    # ---------------- Predict on val and test with the best saved head ----------------
    val_scores = predict_all(model, val_patches)
    test_scores = predict_all(model, test_patches)
    threshold = pick_threshold(torch.sigmoid(image_scores(val_scores)), val_labels)   # chosen on VAL only

    all_scores = []
    splits = {
        "val": (val_data, val_masks, val_labels, val_scores),
        "test": (test_data, test_masks, test_labels, test_scores),
    }
    for split in splits:
        data, masks, labels, scores = splits[split]
        results, split_scores = evaluate(data, masks, labels, scores, threshold)
        results.to_csv(f"fracture_{split}_predictions.csv", index=False)
        split_scores["split"] = split
        all_scores.append(split_scores)

        # heatmap pictures: where the model thinks the fracture is (red) + its box (yellow)
        os.makedirs(f"results/heatmaps_{split}", exist_ok=True)
        square_probs = torch.sigmoid(scores).numpy()
        for i in range(len(data)):
            box = None
            if results["predicted"][i] == "fracture":
                box = [results["box_x1"][i], results["box_y1"][i], results["box_x2"][i], results["box_y2"][i]]
            percent = round(results["probability"][i] * 100)
            name = f"{data['label'][i]}_{percent}_{os.path.basename(data['image_path'][i])}.png"
            save_heatmap(data["image_path"][i], square_probs[i], box, f"results/heatmaps_{split}/{name}")

    all_scores = pd.DataFrame(all_scores)
    all_scores.to_csv("fracture_metrics.csv", index=False)
    print("\n" + all_scores.to_string(index=False))
