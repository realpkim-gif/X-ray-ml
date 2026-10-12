import os
import re
import json
import shutil
import hashlib
import pandas as pd
import torch
from PIL import Image
from augmentation import augment

# folder with one subfolder per label, or a csv with columns: image_path, label,
# or a Roboflow COCO export: <split>/_annotations.coco.json + images
INPUT = "Fratura de membro.v5-dataset-final--400-frat-e-800-n-frat-.coco-segmentation"
OUTPUT = "PreprocessedData"   # gets train/ valid/ test/ (images + _annotations.coco.json) like the input, plus the csvs
VAL_SIZE = 0.15
TEST_SIZE = 0.15           # train gets the rest (70%)
AUGMENT_COPIES = 4         # extra augmented copies per train image (0 = off)
MIN_SIZE = 64              # smallest allowed width/height in pixels
MAX_SIZE = 1024            # bigger images are shrunk so their longest side is this (in pixels)
SEED = 42
# animals whose name starts with one of these come from other scanners/sources and are ALL fractures.
# they only go to train: in val/test the model could spot "different scanner" instead of the fracture.
TRAIN_ONLY = ("CR", "img", "RAYSSINHA")

for split in ["train", "valid", "test", "resized"]:
    shutil.rmtree(f"{OUTPUT}/{split}", ignore_errors=True)   # start empty, so no images from an old run stay behind
    os.makedirs(f"{OUTPUT}/{split}", exist_ok=True)
rejected = []              # every image we throw out, with the reason
categories = []            # COCO only: the label list from the json
coco_anns = {}             # COCO only: image_path -> its fracture outlines (masks)


# 1. Load image paths and labels
if INPUT.endswith(".csv"):
    data = pd.read_csv(INPUT)
    folder = os.path.dirname(INPUT)          # image paths in the csv are relative to the csv's folder
    data["image_path"] = [os.path.join(folder, p) if isinstance(p, str) else p for p in data["image_path"]]
elif os.path.exists(f"{INPUT}/train/_annotations.coco.json"):
    # Roboflow COCO export. An image with an outline = fracture, no outline = normal.
    # Roboflow's own train/valid/test split is ignored, we split again below (by animal).
    rows = []
    for split in os.listdir(INPUT):
        json_path = f"{INPUT}/{split}/_annotations.coco.json"
        if not os.path.exists(json_path):
            continue
        with open(json_path) as f:
            coco = json.load(f)
        categories = coco["categories"]
        for image in coco["images"]:
            path = f"{INPUT}/{split}/{image['file_name']}"
            anns = [a for a in coco["annotations"] if a["image_id"] == image["id"]]
            name = os.path.splitext(image["extra"]["name"])[0]   # original name, e.g. "DX000098_4"
            animal = re.sub(r"_\d+$", "", name)                   # "DX000098_4" -> "DX000098" (same animal)
            rows.append({"image_path": path, "label": "fracture" if anns else "normal", "animal": animal})
            coco_anns[path] = anns
    data = pd.DataFrame(rows)
else:
    rows = []
    for label in os.listdir(INPUT):
        folder = os.path.join(INPUT, label)
        if os.path.isdir(folder):
            for file in os.listdir(folder):
                rows.append({"image_path": os.path.join(folder, file), "label": label})
    data = pd.DataFrame(rows)
if "animal" not in data.columns:
    data["animal"] = data["image_path"]      # no animal ids: every image counts as its own animal
print("loaded:", len(data))


# 2. Remove rows with a missing path or label
data = data.dropna(subset=["image_path", "label"])
data["label"] = data["label"].astype(str).str.strip()
data = data[data["label"] != ""]


# 3. One row per image. An image with several findings gets them joined: "fracture|tumor"
def join_labels(labels):
    findings = "|".join(labels).split("|")
    return "|".join(sorted(set(f.strip() for f in findings if f.strip())))

data = data.groupby("image_path", as_index=False).agg({"label": join_labels, "animal": "first"})
data["original"] = data["image_path"]       # where the image came from (used to find its COCO outlines)
data["augmented"] = False


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


# 9. Split into train / val / test (70 / 15 / 15) by ANIMAL, not by image:
#    all X-rays of one animal go to the same split, so test only has animals the model never saw.
#    Animals are grouped by their findings ("fracture", "normal", "fracture|normal") and each group
#    is split on its own so every split gets the same mix. Train always keeps at least 1 animal per group.
animals = data.groupby("animal", as_index=False)["label"].agg(join_labels)   # one row per animal
train_only = animals["animal"].str.startswith(TRAIN_ONLY)
train_animals, val_animals, test_animals = list(animals[train_only]["animal"]), [], []
animals = animals[~train_only]
for _, group in animals.groupby("label"):
    names = list(group["animal"].sample(frac=1, random_state=SEED))   # shuffle
    n = len(names)
    n_test = min(round(n * TEST_SIZE), n - 1)
    n_val = min(round(n * VAL_SIZE), n - 1 - n_test)
    test_animals += names[:n_test]
    val_animals += names[n_test:n_test + n_val]
    train_animals += names[n_test + n_val:]
train = data[data["animal"].isin(train_animals)].copy()
val = data[data["animal"].isin(val_animals)].copy()
test = data[data["animal"].isin(test_animals)].copy()


# 10. Copy the kept images into OUTPUT/train, valid, test (same layout as the input)
for split, part in [("train", train), ("valid", val), ("test", test)]:
    new_paths = []
    for path in part["image_path"]:
        new_path = f"{OUTPUT}/{split}/{os.path.basename(path)}"
        shutil.copy(path, new_path)
        new_paths.append(new_path)
    part["image_path"] = new_paths


# 11. Augment train images only (val and test must stay real X-rays)
torch.manual_seed(SEED)
transform = augment()
copies = []
for _, row in train.iterrows():
    img = Image.open(row["image_path"]).convert("RGB")
    for k in range(AUGMENT_COPIES):
        path = f"{OUTPUT}/train/aug_{row['md5']}_{k}.png"
        transform(img).save(path)
        copy = row.copy()
        copy["image_path"] = path
        copy["augmented"] = True
        copies.append(copy)
train = pd.concat([train, pd.DataFrame(copies)])


# 12. Write one _annotations.coco.json per folder, like the input.
#     Every image gets its "label". Real images keep their fracture outlines;
#     augmented copies get none (the outline no longer fits after rotating/shifting).
for split, part in [("train", train), ("valid", val), ("test", test)]:
    coco = {"categories": categories, "images": [], "annotations": []}
    for _, row in part.iterrows():
        image_id = len(coco["images"])
        with Image.open(row["image_path"]) as img:
            width, height = img.size
        coco["images"].append({"id": image_id, "file_name": os.path.basename(row["image_path"]),
                               "width": width, "height": height, "label": row["label"],
                               "animal": row["animal"], "augmented": bool(row["augmented"])})
        if not row["augmented"]:
            for ann in coco_anns.get(row["original"], []):
                coco["annotations"].append({**ann, "id": len(coco["annotations"]), "image_id": image_id})
    with open(f"{OUTPUT}/{split}/_annotations.coco.json", "w") as f:
        json.dump(coco, f)


# 13. Save everything
train.to_csv(OUTPUT + "/train.csv", index=False)
val.to_csv(OUTPUT + "/val.csv", index=False)
test.to_csv(OUTPUT + "/test.csv", index=False)
pd.concat(rejected).to_csv(OUTPUT + "/rejected.csv", index=False)
with open(OUTPUT + "/label_map.json", "w") as f:
    json.dump(label_map, f, indent=2)

print("labels:", label_map)
print("train:", len(train), "val:", len(val), "test:", len(test), "rejected:", len(pd.concat(rejected)))
