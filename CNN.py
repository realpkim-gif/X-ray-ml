import torch                       # PyTorch: tensors (multi-dimensional arrays) + automatic gradients
from torch import nn               # nn = layers with learnable weights (Conv2d, Linear, BatchNorm2d, ...)
import torch.nn.functional as F    # F = plain functions with no weights (relu, softmax, ...)

# Shapes in PyTorch are [batch, channels, height, width].
#   batch    = how many images are processed at once (written as B in the shape comments below;
#              not the same as the "B grid" = Blue in the conv1 kernel diagram)
#   channels = how many maps are stacked. ONLY the input is RGB (3); after conv1 every
#              channel is one learned kernel's map (16, 32, ... 512 at the end)
#
# conv1 turns the 3 colors into 16 maps. From then on, the network only sees those
# 16 maps, and color no longer exists anywhere:
#
#   RGB image  [3] -> conv1 -> [16] -> conv2 -> [32] -> ... -> [512]
#   gray image [1] -> conv1 -> [16] -> conv2 -> [32] -> ... -> [512]
#                       ^
#              the only layer that differs
#
# So RGB is NOT an extra 3x in every layer: it only changes conv1's weights
# (16x3x3x3 + 16 = 448 vs 16x1x3x3 + 16 = 160), i.e. 288 of ~2.09M weights.
#
# What the channels hold, layer by layer (3 CHANNELS for RGB, not 3 layers):
#
#   input:   3 channels  (R, G, B)                      <- the only time color exists
#   conv1:  16 channels  (edges, light/dark changes)    <- learned patterns, not colors
#   conv2:  32 channels  (corners, textures, lines)     <- combos of conv1's patterns
#   conv3:  64 channels  (curves, bone edges)
#   ...
#   conv6: 512 channels  (whole structures, e.g. a break in the bone wall)
#
# Every new kernel looks at ALL channels of the previous layer and combines them
# into one new pattern map, so patterns get more complex as you go deeper. Meanwhile
# pooling shrinks the image, so each deep pattern covers a bigger area of the X-ray.
# (The pattern names above are typical examples; training decides what each kernel learns.)
#
# Two separate things in every layer's output:
#   channels       = WHAT pattern  (16, 32, ... 512 kinds of detectors)
#   height x width = WHERE in the image (1024 -> 512 -> ... -> 16, shrunk by pooling)
# Channels are not "scales": scale/size changes only through pooling.


class ConvNet(nn.Module):              # every PyTorch model inherits from nn.Module (tracks weights, train/eval mode)

    def __init__(self, num_classes, dropout=0.3):     # __init__ CREATES the layers; nothing runs yet
        super(ConvNet, self).__init__()                  # set up nn.Module internals; must be the first line

        # ---------------- Stage 1: 1024x1024x3 -> 512x512x16 ----------------
        # Conv2d(in_channels, out_channels, kernel_size, padding)
        #   slides a 3x3 window over the image; each of the 16 filters learns one pattern
        #   (edges, bright/dark changes...). padding=1 adds a 1-pixel border so the size stays 1024.
        #
        # How ONE kernel works (out_channels = how many kernels = how many output maps):
        #   A kernel has one 3x3 grid PER INPUT CHANNEL. conv1 reads R, G, B, so each kernel is
        #   3 grids of 3x3 = 3x3x3 = 27 weights:
        #
        #       R grid       G grid       B grid
        #      [w w w]      [w w w]      [w w w]
        #      [w w w]  +   [w w w]  +   [w w w]   + bias  ->  ONE 3x3 vector
        #      [w w w]      [w w w]      [w w w]
        #
        #   At every position: each grid multiplies its own color's 3x3 patch and sums it,
        #   then the 3 results are ADDED into one number. Sliding over the whole image gives
        #   one 1024x1024 map per kernel. 16 kernels -> 16 maps -> output [B, 16, 1024, 1024].
        #   The colors are merged here: none of the 16 maps is "red" or "blue" anymore.
        #   (X-rays are gray copied into 3 identical channels; Conv2d(1, 16, ...) would also work.)
        #
        #   Same rule in every layer: kernel depth = that layer's in_channels.
        #     conv1 kernel: 3x3x3   (3 grids, one per color)    conv1.weight.shape = [16, 3, 3, 3]
        #     conv2 kernel: 3x3x16  (16 grids, one per map)     conv2.weight.shape = [32, 16, 3, 3]
        #     conv6 kernel: 3x3x256 (256 grids, one per map)    conv6.weight.shape = [512, 256, 3, 3]
        #   Each kernel always produces ONE map, so out_channels = number of output maps.
        #   Weights per layer = out_channels x in_channels x 3 x 3 (+ 1 bias per kernel).
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)   # 3 channels in (RGB) -> 16 kernels -> 16 maps out
        self.bn1 = nn.BatchNorm2d(16)                 # rescales each of the 16 channels to mean 0 / std 1: stable, faster training
        self.lin1 = nn.Conv2d(16, 16, 1)              # 1x1 conv = a Linear layer applied to every pixel: mixes the 16 channels

        # ---------------- Stage 2: 512 -> 256, 16 -> 32 channels ----------------
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)  # 32 kernels, each 3x3x16 (one grid per conv1 map) -> 32 maps
        self.bn2 = nn.BatchNorm2d(32)                 # normalise the 32 channels
        self.lin2 = nn.Conv2d(32, 32, 1)              # per-pixel linear mix of the 32 channels

        # ---------------- Stage 3: 256 -> 128, 32 -> 64 channels ----------------
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)  # widen: 32 -> 64
        self.bn3 = nn.BatchNorm2d(64)                 # normalise
        self.lin3 = nn.Conv2d(64, 64, 1)              # per-pixel linear mix

        # ---------------- Stage 4: 128 -> 64, 64 -> 128 channels ----------------
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1) # widen: 64 -> 128
        self.bn4 = nn.BatchNorm2d(128)                # normalise
        self.lin4 = nn.Conv2d(128, 128, 1)            # per-pixel linear mix

        # ---------------- Stage 5: 64 -> 32, 128 -> 256 channels ----------------
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)  # widen: 128 -> 256 (now bone-level shapes)
        self.bn5 = nn.BatchNorm2d(256)                  # normalise
        self.lin5 = nn.Conv2d(256, 256, 1)              # per-pixel linear mix

        # ---------------- Stage 6: 32 -> 16, 256 -> 512 channels ----------------
        self.conv6 = nn.Conv2d(256, 512, 3, padding=1)  # widen: 256 -> 512 (complex findings, e.g. a cortex break)
        self.bn6 = nn.BatchNorm2d(512)                  # normalise
        self.lin6 = nn.Conv2d(512, 512, 1)              # per-pixel linear mix

        # ---------------- Shared layers ----------------
        self.pool = nn.MaxPool2d(2, 2)              # 2x2 window, step 2: keeps the max of every 2x2 block -> halves height & width
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # averages each channel over all 16x16 positions: 16x16x512 -> 1x1x512
        self.dropout = nn.Dropout(dropout)          # in training, randomly zeroes 30% of values so the model can't rely on a few

        # ---------------- Head: 512-number vector -> one score per finding ----------------
        self.fc1 = nn.Linear(512, 256)              # fully connected: every input connected to every output, 512 -> 256
        self.fc2 = nn.Linear(256, 128)              # 256 -> 128
        self.fc3 = nn.Linear(128, num_classes)      # 128 -> one raw score (logit) per finding

    def stage(self, x, conv, bn, lin):              # one stage = conv -> BN -> ReLU -> 1x1 -> ReLU -> pool
        x = F.relu(bn(conv(x)))                     # 3x3 conv, normalise, ReLU (turns negatives into 0 = non-linearity)
        x = F.relu(lin(x))                          # 1x1 "linear" mix of channels, then ReLU again
        return self.pool(x)                         # halve height and width

    def features(self, x):                          # the "body": image -> feature map
        x = self.stage(x, self.conv1, self.bn1, self.lin1)  # [B,   3, 1024, 1024] -> [B,  16, 512, 512]
        x = self.stage(x, self.conv2, self.bn2, self.lin2)  # [B,  16,  512,  512] -> [B,  32, 256, 256]
        x = self.stage(x, self.conv3, self.bn3, self.lin3)  # [B,  32,  256,  256] -> [B,  64, 128, 128]
        x = self.stage(x, self.conv4, self.bn4, self.lin4)  # [B,  64,  128,  128] -> [B, 128,  64,  64]
        x = self.stage(x, self.conv5, self.bn5, self.lin5)  # [B, 128,   64,   64] -> [B, 256,  32,  32]
        x = self.stage(x, self.conv6, self.bn6, self.lin6)  # [B, 256,   32,   32] -> [B, 512,  16,  16]
        return x #[B, 512,  16,  16]

    def forward(self, x):                           # forward RUNS the layers; called by model(x)
        x = self.features(x)                        # image -> [B, 512, 16, 16] feature map
        x = self.global_pool(x)                     # average each channel -> [B, 512, 1, 1]
        x = torch.flatten(x, 1)                     # drop the 1x1 dims (keep batch dim 0) -> [B, 512]
        x = self.dropout(F.relu(self.fc1(x)))       # 512 -> 256, ReLU, dropout -> [B, 256]
        x = F.relu(self.fc2(x))                     # 256 -> 128, ReLU -> [B, 128]
        x = self.fc3(x)                             # 128 -> num_classes raw scores (no ReLU/sigmoid here)
        return x                                    # train with nn.BCEWithLogitsLoss (it applies sigmoid itself);
                                                    # at prediction time: torch.sigmoid(x) -> probability per finding

