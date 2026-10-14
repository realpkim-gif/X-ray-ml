import os
import random
import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import datasets, transforms
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import FractureLocator as fl     # the patch head: it finds the yellow box, used to make the zoomed X-ray pieces

# ======================================================================================
# The "X-ray or not" GATE
#
# A small, fast check that runs BEFORE the big fracture model (DINO):
#   the gate says "X-ray"     -> run the fracture model
#   the gate says "not X-ray" -> the website asks "We believe this isn't an X-ray. Still run?"
#
# How it works (same idea as Pretrained_CNN.py + main.py):
#   1. ResNet18 (already trained by others, FROZEN = never changes) turns every picture into 512 numbers
#   2. we train a small "head" that looks at those 512 numbers and says: X-ray or not
#
# Pictures used, the SAME number of X-ray and not-X-ray pictures (so the gate isn't biased):
#   X-ray (label 1):
#       - the X-rays of PreprocessedData (same train / val / test split as the fracture model)
#       - zoomed pieces of those X-rays (cut out with the yellow box), saved in xray_crops/
#       - X-ray variations made by gate_extra.py (rotated, inverted, ...), in gate_extra/1/
#   NOT X-ray (label 0):
#       - screenshots, documents, charts, ... made by gate_extra.py, in gate_extra/0/
#       - everyday photos: Caltech-256 (objects) and Places365 (scenes), half of them turned gray
#
# Every time it runs, the scores are added as new rows to gate_scores.csv
#
#   python Xray_Or_Not.py
# ======================================================================================

# ---------- settings ----------
PHOTOS_FOLDER = "gate_data"        # where Caltech-256 and Places365 are downloaded
CROPS_FOLDER = "xray_crops"        # where the zoomed X-ray pieces are saved
EXTRA_FOLDER = "gate_extra"        # pictures made by gate_extra.py
SCORES_FILE = "gate_scores.csv"    # every run adds its scores here
USED_FILE = "gate_used_photos.csv" # which photos were used (so tests can pick photos the gate never saw)
MIN_CROP = 128                     # zoomed pieces are at least this many pixels wide
GRAY_SHARE = 0.5                   # half of the photos are turned gray
EPOCHS = 30
BATCH_SIZE = 64
LEARNING_RATE = 0.001
SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   # the GPU if there is one


# ======================================================================================
# 1. ResNet18: picture -> 512 numbers
# ======================================================================================

# load ResNet18, already trained on ImageNet
resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
# its last layer sorts pictures into 1000 ImageNet classes. We don't want that, we want the 512 numbers
# that go INTO that layer. nn.Identity() = "pass the numbers through unchanged".
resnet.fc = nn.Identity()
resnet = resnet.to(device)
resnet.eval()                                  # inference mode
# FREEZE: none of ResNet18's numbers will ever change
for param in resnet.parameters():
    param.requires_grad = False

# how every picture is prepared for ResNet18:
#   squash it to 224 x 224 (ResNet18's size, no cropping, so nothing gets cut off)
#   turn it into numbers 0-1
#   normalise with the ImageNet averages (what ResNet18 was trained with)
prepare = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def picture_to_vector(picture):
    # one PIL picture -> 512 numbers (a numpy array)
    picture = picture.convert("RGB")                     # 3 colour channels, also for gray pictures
    tensor = prepare(picture)                            # [3, 224, 224]
    tensor = tensor.unsqueeze(0)                         # [1, 3, 224, 224]  (a "batch" of 1 picture)
    tensor = tensor.to(device)
    with torch.no_grad():                                # no training here
        vector = resnet(tensor)                          # [1, 512]
    return vector[0].cpu().numpy()                       # [512]


# ======================================================================================
# 2. The small head: 512 numbers -> "not X-ray" / "X-ray"
# ======================================================================================

class GateHead(nn.Module):
    def __init__(self):
        super(GateHead, self).__init__()
        self.linear1 = nn.Linear(512, 128)   # 512 numbers -> 128 numbers
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)       # randomly switch off 30% while training (less overfitting)
        self.linear2 = nn.Linear(128, 2)     # 128 numbers -> 2 scores: [not X-ray, X-ray]

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x                             # raw scores; softmax turns them into chances


def xray_chance(head, picture):
    # one picture -> the chance (0 to 1) that it is an X-ray. Used by gate_challenge.py too.
    vector = torch.tensor(picture_to_vector(picture)).unsqueeze(0).to(device)   # [1, 512]
    head.eval()
    with torch.no_grad():
        scores = head(vector)                             # [1, 2]
        chances = torch.softmax(scores, dim=1)            # [1, 2], adds up to 1
    return chances[0, 1].item()                           # the "X-ray" chance


def load_head():
    # the trained head, ready to use
    head = GateHead().to(device)
    head.load_state_dict(torch.load("gate_head.pt", map_location=device))
    head.eval()
    return head


# ======================================================================================
# Everything below only runs when you run THIS file (not when gate_challenge.py imports from it)
# ======================================================================================
if __name__ == "__main__":

    # ---------- 3a. zoomed X-ray pieces (made once, saved in xray_crops/) ----------
    def make_crops(split):
        # every real X-ray of this split -> the yellow box from the patch head -> cut out and save
        os.makedirs(f"{CROPS_FOLDER}/{split}", exist_ok=True)
        data, patches, masks, labels = fl.load(split, keep_flipped=False)
        patch_head = fl.PatchHead(768).to(device)
        patch_head.load_state_dict(torch.load("fracture_head.pt", map_location=device))
        heatmaps = torch.sigmoid(fl.predict_all(patch_head, patches)).numpy()     # [N, 32, 32]

        for i in range(len(data)):
            x1, y1, x2, y2 = fl.find_box(heatmaps[i])      # the yellow box, in pixels of the 1024 x 1024 X-ray
            # a box smaller than MIN_CROP is made bigger around its middle
            if x2 - x1 < MIN_CROP:
                middle = (x1 + x2) / 2
                x1 = max(0, middle - MIN_CROP / 2)
                x2 = min(1024, middle + MIN_CROP / 2)
            if y2 - y1 < MIN_CROP:
                middle = (y1 + y2) / 2
                y1 = max(0, middle - MIN_CROP / 2)
                y2 = min(1024, middle + MIN_CROP / 2)
            xray = Image.open(data["image_path"][i])
            crop = xray.crop((round(x1), round(y1), round(x2), round(y2)))
            crop.save(f"{CROPS_FOLDER}/{split}/{os.path.basename(data['image_path'][i])}.png")

    for split in ["train", "val", "test"]:
        if not os.path.exists(f"{CROPS_FOLDER}/{split}"):
            make_crops(split)
            print("made zoomed X-ray pieces for", split)

    # ---------- 3b. the photo datasets ----------
    caltech = datasets.Caltech256(root=PHOTOS_FOLDER, download=True)
    places = datasets.Places365(root=f"{PHOTOS_FOLDER}/places365", split="val", small=True, download=True)
    # every photo number in a random order; we take photos from the end of these lists, each one only once
    caltech_numbers = list(range(len(caltech)))
    random.shuffle(caltech_numbers)
    places_numbers = list(range(len(places)))
    random.shuffle(places_numbers)

    # ---------- 3c. which pictures go into which split ----------
    # every picture is a small list: [source, where to find it, label]
    #   where = a file path, or a photo number for caltech256 / places365
    pictures = {"train": [], "val": [], "test": []}
    used_photos = []

    for split in ["train", "val", "test"]:
        # X-rays (train includes the augmented copies made by preprocess.py)
        data = pd.read_csv(f"PreprocessedData/{split}.csv")
        for path in data["image_path"]:
            pictures[split].append(["xray", path, 1])

        # zoomed X-ray pieces
        for file in sorted(os.listdir(f"{CROPS_FOLDER}/{split}")):
            pictures[split].append(["xray_crop", f"{CROPS_FOLDER}/{split}/{file}", 1])

        # pictures made by gate_extra.py: gate_extra/<label>/<type>/<split>/
        for label in [0, 1]:
            label_folder = f"{EXTRA_FOLDER}/{label}"
            if not os.path.exists(label_folder):
                continue
            for kind in sorted(os.listdir(label_folder)):
                folder = f"{label_folder}/{kind}/{split}"
                if not os.path.exists(folder):
                    continue
                for file in sorted(os.listdir(folder)):
                    pictures[split].append([kind, f"{folder}/{file}", label])

        # photos fill up the "not X-ray" side until both sides have the same number of pictures
        xray_count = 0
        not_xray_count = 0
        for picture in pictures[split]:
            if picture[2] == 1:
                xray_count = xray_count + 1
            else:
                not_xray_count = not_xray_count + 1
        photos_needed = max(0, xray_count - not_xray_count)
        for i in range(photos_needed):
            if i % 2 == 0:                                  # every other photo from each dataset
                number = caltech_numbers.pop()
                pictures[split].append(["caltech256", number, 0])
                used_photos.append(["caltech256", number])
            else:
                number = places_numbers.pop()
                pictures[split].append(["places365", number, 0])
                used_photos.append(["places365", number])

    pd.DataFrame(used_photos, columns=["source", "number"]).to_csv(USED_FILE, index=False)

    # ---------- 3d. every picture -> 512 numbers ----------
    vectors = {}
    labels = {}
    sources = {}
    for split in ["train", "val", "test"]:
        split_vectors = []
        split_labels = []
        split_sources = []
        for source, where, label in pictures[split]:
            if source == "caltech256":
                picture = caltech[where][0]                 # caltech[number] = (picture, class)
            elif source == "places365":
                picture = places[where][0]
            else:
                picture = Image.open(where)                 # a file

            # half of the photos turn gray, so "gray" alone can't mean "X-ray"
            if source == "caltech256" or source == "places365":
                if random.random() < GRAY_SHARE:
                    picture = picture.convert("L")

            split_vectors.append(picture_to_vector(picture))
            split_labels.append(label)
            split_sources.append(source)

        vectors[split] = torch.tensor(np.array(split_vectors), dtype=torch.float32).to(device)   # [N, 512]
        labels[split] = torch.tensor(split_labels, dtype=torch.long).to(device)                  # [N]
        sources[split] = split_sources
        print(split, len(split_labels), "pictures:", sum(split_labels), "X-ray and",
              len(split_labels) - sum(split_labels), "not X-ray")

    # ======================================================================================
    # 4. Train the head
    # ======================================================================================
    head = GateHead().to(device)
    loss_function = nn.CrossEntropyLoss()          # both sides have the same number of pictures, so no weights
    optimizer = torch.optim.AdamW(head.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    best_val_loss = 999999

    for epoch in range(EPOCHS):
        # training: go through the train pictures in a random order, BATCH_SIZE at a time
        head.train()                                    # dropout on
        order = torch.randperm(len(labels["train"]))
        losses = []
        for start in range(0, len(order), BATCH_SIZE):
            batch = order[start:start + BATCH_SIZE]
            optimizer.zero_grad()                        # forget the previous step
            scores = head(vectors["train"][batch])       # [B, 2]
            loss = loss_function(scores, labels["train"][batch])
            loss.backward()                              # work out how to improve
            optimizer.step()                             # improve the head a little
            losses.append(loss.item())
        train_loss = sum(losses) / len(losses)

        # validation: how good is it on pictures it doesn't train on?
        head.eval()                                     # dropout off
        with torch.no_grad():
            val_loss = loss_function(head(vectors["val"]), labels["val"]).item()
        if val_loss < best_val_loss:                    # the best epoch so far -> save it
            best_val_loss = val_loss
            torch.save(head.state_dict(), "gate_head.pt")
        print(f"Epoch [{epoch + 1}/{EPOCHS}], Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}")

    print("best val loss:", round(best_val_loss, 4), "saved to gate_head.pt")

    # ======================================================================================
    # 5. Scores with the best head, saved to gate_scores.csv
    #    precision = when it says "X-ray", how often it really is one
    #    recall    = of the real X-rays, how many it lets through
    # ======================================================================================
    head = load_head()
    time_now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    score_rows = []
    for split in ["val", "test"]:
        with torch.no_grad():
            chances = torch.softmax(head(vectors[split]), dim=1)[:, 1].cpu().numpy()   # X-ray chance per picture
        truth = labels[split].cpu().numpy()
        predicted = []
        for chance in chances:
            if chance >= 0.5:
                predicted.append(1)
            else:
                predicted.append(0)
        predicted = np.array(predicted)

        # which kinds of pictures did it get wrong?
        wrong_by_source = {}
        for i in range(len(truth)):
            if predicted[i] != truth[i]:
                wrong_by_source[sources[split][i]] = wrong_by_source.get(sources[split][i], 0) + 1

        row = {
            "time": time_now,
            "split": split,
            "pictures": len(truth),
            "accuracy": round(accuracy_score(truth, predicted), 4),
            "precision": round(precision_score(truth, predicted), 4),
            "recall": round(recall_score(truth, predicted), 4),
            "f1": round(f1_score(truth, predicted), 4),
            "auroc": round(roc_auc_score(truth, chances), 4),
            "mistakes": int((predicted != truth).sum()),
            "wrong_by_source": str(wrong_by_source),
        }
        score_rows.append(row)
        print(row)

    # add the new rows to gate_scores.csv (make the file if it isn't there yet)
    new_scores = pd.DataFrame(score_rows)
    if os.path.exists(SCORES_FILE):
        new_scores = pd.concat([pd.read_csv(SCORES_FILE), new_scores])
    new_scores.to_csv(SCORES_FILE, index=False)
    print("scores added to", SCORES_FILE)
