import os
import json
import math
import torch
from torch import nn
import pandas as pd
import numpy as np
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
# ---------- always work from the project folder (X-ray-ml) ----------
# This file lives in src/<folder>/, but all paths below ("data/...", "src/models/weights/...") start at the
# project folder. These lines go there first, so it works from PyCharm's Run button AND from a terminal.
PROJECT_FOLDER = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # up 3: file -> src/<folder> -> src -> project
os.chdir(PROJECT_FOLDER)


device = "cuda" if torch.cuda.is_available() else "cpu"
DATA = "data/PreprocessedData"   # where preprocess.py writes the csvs and label_map.json
EPOCHS = 50
BATCH_SIZE = 8
TARGET_RECALL = 0.90  # per finding, pick the threshold (on val) that still catches 90% of real cases
WARMUP = 0.05      # first 5% of training steps: lr climbs from 0 up to its full value

with open(f"{DATA}/label_map.json") as f:
  label_map = json.load(f)               # {"fracture": 0, "normal": 1, ...}
num_classes = len(label_map)
# the label names in number order: class_names[0] = name of label 0, class_names[1] = name of label 1, ...
class_names = []
for number in range(num_classes):
  for name in label_map:
    if label_map[name] == number:
      class_names.append(name)


def load(split):
  cnn = torch.tensor(np.load(f"data/features/cnn_{split}.npy"), dtype=torch.float32)    # [N, 1024]
  dino = torch.tensor(np.load(f"data/features/dino_{split}.npy"), dtype=torch.float32)  # [N, 1536]

  # targets: one row per image, a 1 for every finding it has. "1|5" -> [0, 1, 0, 0, 0, 1, 0]
  label_ids = pd.read_csv(f"{DATA}/{split}.csv", dtype={"label_ids": str})["label_ids"]
  targets = torch.zeros(len(label_ids), num_classes) #create target matrix
  for i, ids in enumerate(label_ids):
    for j in ids.split("|"):
      targets[i, int(j)] = 1
  return cnn.to(device), dino.to(device), targets.to(device)


class Head(nn.Module):
  def __init__(self, cnn_size, dino_size, num_classes):
    super(Head, self).__init__()
    # LayerNorm puts each vector on the same scale first, so CNN and DINO count equally
    self.norm1 = nn.LayerNorm(cnn_size)
    self.norm2 = nn.LayerNorm(dino_size)
    self.fc1 = nn.Linear(cnn_size, 512)    # CNN vector  [B, 1024] -> [B, 512]
    self.fc2 = nn.Linear(dino_size, 512)   # DINO vector [B, 1536] -> [B, 512]
    self.relu = nn.ReLU()      # Activation function
    self.dropout = nn.Dropout(0.3)    # randomly zero 30% while training (less overfitting)
    self.fc3 = nn.Linear(512, num_classes)  # one raw score per finding

  def forward(self, cnn, dino):
    y = self.fc1(self.norm1(cnn))    # [B, 512]
    z = self.fc2(self.norm2(dino))   # [B, 512]
    x = z + y            # same size now, so they can be added
    x = self.relu(x)
    x = self.dropout(x)
    x = self.fc3(x)      # [B, num_classes] raw scores
    return x             # raw scores: BCEWithLogitsLoss applies sigmoid itself while training


def make_criterion(train_targets):
  # class imbalance: few positives -> each missed positive costs more.
  # pos_weight = negatives / positives per finding (e.g. 800 normal / 400 fracture = 2)
  positives = train_targets.sum(0)              # how many images have each finding  [num_classes]
  positives = positives.clamp(min=1)            # at least 1, so we never divide by 0
  negatives = len(train_targets) - positives    # how many images DON'T have it
  pos_weight = negatives / positives
  return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def make_scheduler(optimizer, total_steps):
  # linear warmup: lr goes 0 -> full over the first WARMUP part of training
  # then cosine decay: lr slides smoothly from full down to 0 by the last step
  warmup_steps = max(1, int(WARMUP * total_steps))

  # LambdaLR calls this function at every step and multiplies the base lr by what it returns
  def lr_factor(step):
    if step < warmup_steps:
      return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)   # 0 -> 1
    return 0.5 * (1 + math.cos(math.pi * progress))

  return optim.lr_scheduler.LambdaLR(optimizer, lr_factor)


def pick_thresholds(probs, targets):
  # one threshold per finding, chosen on VAL (never on test, that would be cheating).
  # the highest threshold that still catches TARGET_RECALL of the real cases.
  probs = probs.cpu().numpy()
  targets = targets.cpu().numpy()
  thresholds = []
  for c in range(num_classes):
    # the model's scores for the images that really have finding c, sorted low -> high
    positive_probs = []
    for i in range(len(targets)):
      if targets[i, c] == 1:
        positive_probs.append(probs[i, c])
    positive_probs = np.sort(positive_probs)
    if len(positive_probs) == 0:
      thresholds.append(0.5)
      continue
    allowed_misses = int(len(positive_probs) * (1 - TARGET_RECALL))  # e.g. 10% may fall below
    thresholds.append(float(positive_probs[allowed_misses]))
  return thresholds


def metrics(probs, targets, thresholds):
  # one row per finding: AUROC (threshold-free) + accuracy/precision/recall/f1 at the chosen threshold
  probs = probs.cpu().numpy()
  targets = targets.cpu().numpy()
  rows = []
  for c in range(num_classes):
    y = targets[:, c]                          # truth for finding c (1 or 0 per image)
    p = probs[:, c]                            # the model's chance for finding c
    predicted = (p >= thresholds[c]).astype(int)
    # AUROC needs both kinds (some images with the finding and some without)
    if 0 < y.sum() < len(y):
      auroc = roc_auc_score(y, p)
    else:
      auroc = float("nan")
    rows.append([class_names[c], round(thresholds[c], 3), round(auroc, 3),
                 round(recall_score(y, predicted, zero_division=0), 3),
                 round(precision_score(y, predicted, zero_division=0), 3),
                 round(f1_score(y, predicted, zero_division=0), 3),
                 round(accuracy_score(y, predicted), 3), int(y.sum())])
  return pd.DataFrame(rows, columns=["finding", "threshold", "auroc", "recall", "precision", "f1", "accuracy", "positives"])


def predict(split, outputs, thresholds):
  # outputs = the model's raw scores [N, num_classes], so any model (CNN+DINO or DINO only) can use this
  probs = torch.sigmoid(outputs)               # each finding 0-1

  data = pd.read_csv(f"{DATA}/{split}.csv", dtype={"label_ids": str})  # keep "1|5" as text
  rows = []
  for i in range(len(data)):
    # every finding with probability above its own threshold counts as predicted
    predicted = []
    for number in range(num_classes):
      if probs[i, number] >= thresholds[number]:
        predicted.append(str(number))
    rows.append([data["image_path"][i], data["label_ids"][i], "|".join(predicted)])   # e.g. "1|5" vs "1"

  results = pd.DataFrame(rows, columns=["image_path", "true", "predicted"])
  for number in range(num_classes):            # one column per finding with its probability
    results[str(number)] = probs[:, number].cpu().numpy().round(3)   # columns "0", "1", ...
  return results


def evaluate(name, val_outputs, val_targets, test_outputs, test_targets):
  # thresholds from val, then score val and test with them. name = file prefix ("" or "dino_")
  thresholds = pick_thresholds(torch.sigmoid(val_outputs), val_targets)
  thresholds_by_name = {}                    # e.g. {"fracture": 0.084, "normal": 0.755}
  for number in range(num_classes):
    thresholds_by_name[class_names[number]] = thresholds[number]
  with open(f"src/models/weights/{name}thresholds.json", "w") as f:
    json.dump(thresholds_by_name, f, indent=2)

  for split, outputs, targets in [("val", val_outputs, val_targets), ("test", test_outputs, test_targets)]:
    predict(split, outputs, thresholds).to_csv(f"metrics/{name}{split}_predictions.csv", index=False)
    scores = metrics(torch.sigmoid(outputs), targets, thresholds)
    scores.to_csv(f"metrics/{name}{split}_metrics.csv", index=False)
    print(f"\n{split} metrics:\n{scores.to_string(index=False)}")


# everything below only runs when you run main.py itself, not when another file imports from it
if __name__ == "__main__":
  train_cnn, train_dino, train_targets = load("train")
  val_cnn, val_dino, val_targets = load("val")
  test_cnn, test_dino, test_targets = load("test")

  # hands out the train data in shuffled batches: (cnn, dino, targets) per batch
  train_loader = DataLoader(TensorDataset(train_cnn, train_dino, train_targets),
                            batch_size=BATCH_SIZE, shuffle=True)

  model = Head(train_cnn.shape[1], train_dino.shape[1], num_classes).to(device)
  criterion = make_criterion(train_targets)   # one yes/no loss per finding, weighted for imbalance
  optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)  # weight_decay = less overfitting
  scheduler = make_scheduler(optimizer, EPOCHS * len(train_loader))   # changes lr every batch
  best_val_loss = float("inf")

  trainHistory = []      # one row per epoch: [epoch, train_loss, val_loss]

  for epoch in range(EPOCHS):
    # training
    model.train()                               # dropout on
    batch_losses = []
    for batch, (cnn, dino, targets) in enumerate(train_loader):   # one batch of BATCH_SIZE images
      optimizer.zero_grad()                     # Clear previous gradients
      outputs = model(cnn, dino)                # Forward pass
      loss = criterion(outputs, targets)        # Calculate loss
      loss.backward()                           # Backward pass to compute gradients
      optimizer.step()                          # Update weights
      scheduler.step()                          # move lr one step along warmup/decay

      batch_losses.append(loss.item())
    train_loss = sum(batch_losses) / len(batch_losses)    # average loss over all batches

    # validation
    model.eval()                                # dropout off
    with torch.no_grad():
      val_loss = criterion(model(val_cnn, val_dino), val_targets).item()
    trainHistory.append([epoch + 1, train_loss, val_loss])

    if val_loss < best_val_loss:                # keep the epoch that did best on val
      best_val_loss = val_loss
      torch.save(model.state_dict(), "src/models/weights/head.pt")
    print(f'Epoch [{epoch + 1}/{EPOCHS}], Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}, lr: {scheduler.get_last_lr()[0]:.6f}')

  print("best val loss:", best_val_loss, "saved to models/head.pt")

  # lists -> tables, saved so you can plot them later
  trainHistory = pd.DataFrame(trainHistory, columns=["epoch", "train_loss", "val_loss"])
  trainHistory.to_csv("metrics/trainHistory.csv", index=False)


  # ---------------- Predict on val and test with the best saved head ----------------
  model.load_state_dict(torch.load("src/models/weights/head.pt"))
  model.eval()

  with torch.no_grad():                         # no gradients needed for predicting
    evaluate("", model(val_cnn, val_dino), val_targets, model(test_cnn, test_dino), test_targets)

