import torch
import torch.nn as nn

# Initialize the loss function
criterion = nn.BCEWithLogitsLoss()

# Simulated model outputs: 3 samples, raw logits (unbounded values)
logits = torch.tensor([2.5, -1.0, 0.3], dtype=torch.float32)

# Ground truth targets: Must be floats between 0 and 1
targets = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)

# Calculate loss
loss = criterion(logits, targets)
print(f"Loss: {loss.item()}")