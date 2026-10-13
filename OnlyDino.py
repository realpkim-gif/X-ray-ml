import torch
from torch import nn
import pandas as pd
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from main import load, evaluate, make_criterion, make_scheduler, device, num_classes, EPOCHS, BATCH_SIZE   # same settings and methods as main.py

# Same as main.py but the head only sees DINO vectors (no CNN).
# Thresholds, metrics, class weighting and lr schedule all come from main.py too.

train_cnn, train_dino, train_targets = load("train")   # load gives cnn too, we just don't use it
val_cnn, val_dino, val_targets = load("val")
test_cnn, test_dino, test_targets = load("test")

# hands out the train ModelWeights in shuffled batches: (dino, targets) per batch
train_loader = DataLoader(TensorDataset(train_dino, train_targets),
                          batch_size=BATCH_SIZE, shuffle=True)

class DinoHead(nn.Module):
  def __init__(self, dino_size, num_classes):
    super(DinoHead, self).__init__()
    self.norm2 = nn.LayerNorm(dino_size)    # same scaling as main.py's head
    self.fc2 = nn.Linear(dino_size, 512)   # DINO vector [B, 1536] -> [B, 512]
    self.relu = nn.ReLU()      # Activation function
    self.dropout = nn.Dropout(0.3)    # randomly zero 30% while training (less overfitting)
    self.fc3 = nn.Linear(512, num_classes)  # one raw score per finding

  def forward(self, dino):
    x = self.fc2(self.norm2(dino))   # [B, 512]
    x = self.relu(x)
    x = self.dropout(x)
    x = self.fc3(x)      # [B, num_classes] raw scores
    return x             # raw scores: BCEWithLogitsLoss applies sigmoid itself while training


model = DinoHead(train_dino.shape[1], num_classes).to(device)
criterion = make_criterion(train_targets)   # one yes/no loss per finding, weighted for imbalance
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)  # weight_decay = less overfitting
scheduler = make_scheduler(optimizer, EPOCHS * len(train_loader))   # same warmup + decay as main.py
best_val_loss = float("inf")

trainHistory = []      # one row per epoch: [epoch, train_loss, val_loss]

for epoch in range(EPOCHS):
  # training
  model.train()                               # dropout on
  batch_losses = []
  for batch, (dino, targets) in enumerate(train_loader):   # one batch of BATCH_SIZE images
    optimizer.zero_grad()                     # Clear previous gradients
    outputs = model(dino)                     # Forward pass
    loss = criterion(outputs, targets)        # Calculate loss
    loss.backward()                           # Backward pass to compute gradients
    optimizer.step()                          # Update weights
    scheduler.step()                          # move lr one step along warmup/decay

    batch_losses.append(loss.item())
  train_loss = sum(batch_losses) / len(batch_losses)    # average loss over all batches

  # validation
  model.eval()                                # dropout off
  with torch.no_grad():
    val_loss = criterion(model(val_dino), val_targets).item()
  trainHistory.append([epoch + 1, train_loss, val_loss])

  if val_loss < best_val_loss:                # keep the epoch that did best on val
    best_val_loss = val_loss
    torch.save(model.state_dict(), "dino_head.pt")   # own file, so main.py's head.pt is not overwritten
  print(f'Epoch [{epoch + 1}/{EPOCHS}], Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}, lr: {scheduler.get_last_lr()[0]:.6f}')

print("best val loss:", best_val_loss, "saved to dino_head.pt")

# lists -> tables, saved so you can plot them later
trainHistory = pd.DataFrame(trainHistory, columns=["epoch", "train_loss", "val_loss"])
trainHistory.to_csv("dino_trainHistory.csv", index=False)


# ---------------- Predict on val and test with the best saved head ----------------
model.load_state_dict(torch.load("dino_head.pt"))
model.eval()

with torch.no_grad():                         # no gradients needed for predicting
  # same thresholds + metrics as main.py, files get a "dino_" prefix
  evaluate("dino_", model(val_dino), val_targets, model(test_dino), test_targets)
