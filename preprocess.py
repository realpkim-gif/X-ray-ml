import os
import json
import hashlib
import pandas as pd
import torch
from PIL import Image
from augmentation import augment

INPUT = "data/raw"         # folder with one subfolder per label, or a csv with columns: image_path, label
OUTPUT = "data/processed"
VAL_SIZE = 0.15
TEST_SIZE = 0.15           # train gets the rest (70%)
AUGMENT_COPIES = 1         # extra augmented copies per train image (0 = off)
MIN_SIZE = 64              # smallest allowed width/height in pixels
MAX_SIZE = 1024            # bigger images are shrunk so their longest side is this (in pixels)
SEED = 42

os.makedirs(OUTPUT + "/augmented", exist_ok=True)
os.makedirs(OUTPUT + "/resized", exist_ok=True)
rejected = []              # every image we throw out, with the reason


# 1. Load image paths and labels
if INPUT.endswith(".csv"):
    data = pd.read_csv(INPUT)
    folder = os.path.dirname(INPUT)          # image paths in the csv are relative to the csv's folder
    data["image_path"] = [os.path.join(folder, p) if isinstance(p, str) else p for p in data["image_path"]]
else:
    rows = []
    for label in os.listdir(INPUT):
        folder = os.path.join(INPUT, label)
        if os.path.isdir(folder):
            for file in os.listdir(folder):
                rows.append({"image_path": os.path.join(folder, file), "label": label})
    data = pd.DataFrame(rows)
print("loaded:", len(data))


# 2. Remove rows with a missing path or label
data = data.dropna(subset=["image_path", "label"])
data["label"] = data["label"].astype(str).str.strip()
data = data[data["label"] != ""]


# 3. One row per image. An image with several findings gets them joined: "fracture|tumor"
def join_labels(labels):
    findings = "|".join(labels).split("|")
    return "|".join(sorted(set(f.strip() for f in findings if f.strip())))

data = data.groupby("image_path", as_index=False)["label"].agg(join_labels)


# 4. Open every image: find broken files, its size, and a hash of its content
def check_image(path):
    try:
        with Image.open(path) as img:
            img.verify()                          # fails on broken / cut-off files
        with Image.open(path) as img:
            size = min(img.size)                  # smallest side in pixels
        with open(path, "rb") as f:
            md5 = hashlib.md5(f.read()).hexdigest()  # same content -> same hash
        return md5, size
    except Exception:
        return None, 0

results = [check_image(path) for path in data["image_path"]]
data["md5"] = [md5 for md5, size in results]
data["size"] = [size for md5, size in results]

broken = data["md5"].isna()
rejected.append(data[broken].assign(reason="broken or missing"))
data = data[~broken]


# 5. Remove duplicate images (same content saved twice).
#    If the copies have different labels we can't trust either, so remove all of them.
conflict = data.groupby("md5")["label"].transform("nunique") > 1
duplicate = data.duplicated("md5") & ~conflict
rejected.append(data[conflict].assign(reason="duplicate with different labels"))
rejected.append(data[duplicate].assign(reason="duplicate"))
data = data[~conflict & ~duplicate]


# 6. Remove images that are too small to be a real X-ray
too_small = data["size"] < MIN_SIZE
rejected.append(data[too_small].assign(reason="too small"))
data = data[~too_small]
print("after cleaning:", len(data))


# 7. Shrink images that are too large. Keeps the shape (no stretching) and saves
#    a smaller copy in OUTPUT/resized; the original file is not touched.
new_paths = []
for _, row in data.iterrows():
    path = row["image_path"]
    with Image.open(path) as img:
        if max(img.size) > MAX_SIZE:
            img.thumbnail((MAX_SIZE, MAX_SIZE), Image.LANCZOS)   # shrink, keep aspect ratio
            path = f"{OUTPUT}/resized/{row['md5']}.png"
            img.save(path)
    new_paths.append(path)
print("resized:", sum(new != old for new, old in zip(new_paths, data["image_path"])))
data["image_path"] = new_paths


# 8. Turn labels into numbers: {"cardiomegaly": 0, "fracture": 1, ...}
classes = sorted(set("|".join(data["label"]).split("|")))
label_map = {name: number for number, name in enumerate(classes)}
data["label_ids"] = [
    "|".join(str(label_map[name]) for name in labels.split("|")) for labels in data["label"]
]


# 9. Split into train / val / test (70 / 15 / 15).
#    Each label is split on its own so every split gets the same mix of labels.
#    Train always keeps at least 1 image, so rare labels are never lost.
train, val, test = [], [], []
for _, group in data.groupby("label"):
    group = group.sample(frac=1, random_state=SEED)   # shuffle
    n = len(group)
    n_test = min(round(n * TEST_SIZE), n - 1)
    n_val = min(round(n * VAL_SIZE), n - 1 - n_test)
    test.append(group[:n_test])
    val.append(group[n_test:n_test + n_val])
    train.append(group[n_test + n_val:])
train, val, test = pd.concat(train), pd.concat(val), pd.concat(test)


# 10. Augment train images only (val and test must stay real X-rays)
torch.manual_seed(SEED)
transform = augment()
copies = []
for _, row in train.iterrows():
    img = Image.open(row["image_path"]).convert("RGB")
    for k in range(AUGMENT_COPIES):
        path = f"{OUTPUT}/augmented/{row['md5']}_{k}.png"
        transform(img).save(path)
        copy = row.copy()
        copy["image_path"] = path
        copies.append(copy)
train = pd.concat([train, pd.DataFrame(copies)])


# 11. Save everything
train.to_csv(OUTPUT + "/train.csv", index=False)
val.to_csv(OUTPUT + "/val.csv", index=False)
test.to_csv(OUTPUT + "/test.csv", index=False)
pd.concat(rejected).to_csv(OUTPUT + "/rejected.csv", index=False)
with open(OUTPUT + "/label_map.json", "w") as f:
    json.dump(label_map, f, indent=2)

print("labels:", label_map)
print("train:", len(train), "val:", len(val), "test:", len(test), "rejected:", len(pd.concat(rejected)))
