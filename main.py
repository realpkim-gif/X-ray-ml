import json
import torch
from torch import nn
import pandas as pd
import numpy as np
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score,f1_score, precision_score, recall_score

device = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 50
BATCH_SIZE = 8
THRESHOLD = 0.75   # probability above this = finding is predicted

with open("data/processed/label_map.json") as f:
  label_map = json.load(f)               # {"cardiomegaly": 0, "fracture": 1, ...}
num_classes = len(label_map)


def load(split):
  cnn = torch.tensor(np.load(f"data/features/cnn_{split}.npy"), dtype=torch.float32)    # [N, 1024]
  dino = torch.tensor(np.load(f"data/features/dino_{split}.npy"), dtype=torch.float32)  # [N, 2048]

  # targets: one row per image, a 1 for every finding it has. "1|5" -> [0, 1, 0, 0, 0, 1, 0]
  label_ids = pd.read_csv(f"data/processed/{split}.csv", dtype={"label_ids": str})["label_ids"]
  targets = torch.zeros(len(label_ids), num_classes) #create target matrix
  for i, ids in enumerate(label_ids):
    for j in ids.split("|"):
      targets[i, int(j)] = 1
  return cnn.to(device), dino.to(device), targets.to(device)


train_cnn, train_dino, train_targets = load("train")
val_cnn, val_dino, val_targets = load("val")
test_cnn, test_dino, test_targets = load("test")

# hands out the train data in shuffled batches of 64 images: (cnn, dino, targets) per batch
train_loader = DataLoader(TensorDataset(train_cnn, train_dino, train_targets),
                          batch_size=BATCH_SIZE, shuffle=True)

class Head(nn.Module):
  def __init__(self, num_classes):
    super(Head, self).__init__()
    self.fc1 = nn.Linear(1024, 512)   # CNN vector  [B, 1024] -> [B, 512]
    self.fc2 = nn.Linear(2048, 512)   # DINO vector [B, 2048] -> [B, 512]
    self.relu = nn.ReLU()      # Activation function
    self.dropout = nn.Dropout(0.3)    # randomly zero 30% while training (less overfitting)
    self.fc3 = nn.Linear(512, num_classes)  # one raw score per finding

  def forward(self, cnn, dino):
    y = self.fc1(cnn)    # [B, 512]
    z = self.fc2(dino)   # [B, 512]
    x = z + y            # same size now, so they can be added
    x = self.relu(x)
    x = self.dropout(x)
    x = self.fc3(x)      # [B, num_classes] raw scores
    return x             # raw scores: BCEWithLogitsLoss applies sigmoid itself while training


model = Head(num_classes).to(device)
criterion = nn.BCEWithLogitsLoss()       # one yes/no loss per finding (multi-label)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)  # weight_decay = less overfitting
best_val_loss = float("inf")

trainHistory = []      # one row per epoch: [epoch, train_loss, val_loss]
trainLogs = []    # one row per batch: [epoch, batch, train_loss]

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

    batch_losses.append(loss.item())
  train_loss = sum(batch_losses) / len(batch_losses)    # average loss over all batches

  # validation
  model.eval()                                # dropout off
  with torch.no_grad():
    val_loss = criterion(model(val_cnn, val_dino), val_targets).item()
  trainHistory.append([epoch + 1, train_loss, val_loss])

  if val_loss < best_val_loss:                # keep the epoch that did best on val
    best_val_loss = val_loss
    torch.save(model.state_dict(), "head.pt")
  print(f'Epoch [{epoch + 1}/{EPOCHS}], Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}')

print("best val loss:", best_val_loss, "saved to head.pt")

# lists -> tables, saved so you can plot them later
trainHistory = pd.DataFrame(trainHistory, columns=["epoch", "train_loss", "val_loss"])
trainHistory.to_csv("trainHistory.csv", index=False)


# ---------------- Predict on val and test with the best saved head ----------------
model.load_state_dict(torch.load("head.pt"))
model.eval()
test_cnn, test_dino, test_targets = load("test")


def predict(split, cnn, dino, targets):
  with torch.no_grad():
    outputs = model(cnn, dino)                 # raw scores [N, num_classes]
  loss = criterion(outputs, targets).item()
  probs = torch.sigmoid(outputs)               # each finding 0-1

  data = pd.read_csv(f"data/processed/{split}.csv", dtype={"label_ids": str})  # keep "1|5" as text
  rows = []
  for i in range(len(data)):
    # every finding with probability above THRESHOLD counts as predicted
    predicted = []
    for number in range(num_classes):
      if probs[i, number] > THRESHOLD:
        predicted.append(str(number))
    rows.append([data["image_path"][i], data["label_ids"][i], "|".join(predicted)])   # e.g. "1|5" vs "1"

  results = pd.DataFrame(rows, columns=["image_path", "true", "predicted"])
  for number in range(num_classes):            # one column per finding with its probability
    results[str(number)] = probs[:, number].cpu().numpy().round(3)   # columns "0", "1", ...

  accuracy = accuracy_score(results["true"], results["predicted"])
  precision = precision_score(results["true"], results["predicted"])
  recall = recall_score(results["true"], results["predicted"])
  f1 = f1_score(results["true"], results["predicted"])

  results[["precision", "recall", "f1", "accuracy"]] = accuracy, precision, recall, f1

  return results


valResults = predict("val", val_cnn, val_dino, val_targets)
valResults.to_csv("val_predictions.csv", index=False)

testResults = predict("test", test_cnn, test_dino, test_targets)
testResults.to_csv("test_predictions.csv", index=False)
