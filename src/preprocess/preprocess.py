import os
import re
import json
import shutil
import hashlib
import pandas as pd
import torch
from PIL import Image
# ---------- always work from the project folder (X-ray-ml) ----------
# This file lives in src/<folder>/, but all paths below ("data/...", "src/models/weights/...") start at the
# project folder. These lines go there first, so it works from PyCharm's Run button AND from a terminal.
PROJECT_FOLDER = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # up 3: file -> src/<folder> -> src -> project
os.chdir(PROJECT_FOLDER)

from augmentation import augment

# ======================================================================================
# preprocess.py - turns the downloaded X-rays into a clean dataset, split into train / val / test
#
# Input (INPUT) can be one of three things:
#   - a Roboflow COCO export: <split>/_annotations.coco.json + images   (what we use now)
#   - a csv with the columns image_path, label
#   - a folder with one subfolder per label
#
# Output (OUTPUT = PreprocessedData/):
#   train/ valid/ test/        the images + one _annotations.coco.json each (same layout as the input)
#   train.csv val.csv test.csv one row per image
#   label_map.json             label name -> number, e.g. {"fracture": 0, "normal": 1}
#   rejected.csv               every image we threw out, and why
# ======================================================================================

# ---------- settings ----------
INPUT = "data/raw/Fratura de membro.v5-dataset-final--400-frat-e-800-n-frat-.coco-segmentation"
OUTPUT = "data/PreprocessedData"
VAL_SIZE = 0.15
TEST_SIZE = 0.15           # train gets the rest (70%)
AUGMENT_COPIES = 4         # extra augmented copies of every train image (0 = none)
MIN_SIZE = 64              # images smaller than this (in pixels) are thrown out
MAX_SIZE = 1024            # images bigger than this are shrunk so their longest side is this
SEED = 42
# animals whose name starts with one of these come from other scanners/sources and are ALL fractures.
# They only go to train: in val/test the model could spot "different scanner" instead of the fracture.
TRAIN_ONLY = ("CR", "img", "RAYSSINHA")

# start with empty output folders, so no images from an old run stay behind
for folder in ["train", "valid", "test", "resized"]:
    shutil.rmtree(f"{OUTPUT}/{folder}", ignore_errors=True)
    os.makedirs(f"{OUTPUT}/{folder}", exist_ok=True)

rejected = []              # tables of thrown-out images, each with a "reason" column
categories = []            # COCO only: the list of label names in the json
outlines = {}              # COCO only: image path -> its fracture outlines


# ======================================================================================
# 1. Load every image path with its label (and its animal)
# ======================================================================================
rows = []

if INPUT.endswith(".csv"):
    # a csv: image paths in it are relative to the csv's own folder
    table = pd.read_csv(INPUT)
    folder = os.path.dirname(INPUT)
    for i in range(len(table)):
        path = table["image_path"][i]
        if isinstance(path, str):
            path = os.path.join(folder, path)
        rows.append({"image_path": path, "label": table["label"][i]})

elif os.path.exists(f"{INPUT}/train/_annotations.coco.json"):
    # Roboflow COCO export: an image WITH an outline = fracture, WITHOUT = normal.
    # Roboflow's own train/valid/test split is ignored: we split again below, by animal.
    for split in os.listdir(INPUT):
        json_path = f"{INPUT}/{split}/_annotations.coco.json"
        if not os.path.exists(json_path):
            continue
        with open(json_path) as f:
            coco = json.load(f)
        categories = coco["categories"]
        for image in coco["images"]:
            path = f"{INPUT}/{split}/{image['file_name']}"

            # this image's outlines
            image_outlines = []
            for annotation in coco["annotations"]:
                if annotation["image_id"] == image["id"]:
                    image_outlines.append(annotation)
            outlines[path] = image_outlines

            if len(image_outlines) > 0:
                label = "fracture"
            else:
                label = "normal"

            # the animal: "DX000098_4.jpg" -> "DX000098_4" -> "DX000098" (all X-rays of one animal)
            name = os.path.splitext(image["extra"]["name"])[0]
            animal = re.sub(r"_\d+$", "", name)      # remove the "_<number>" at the end
            rows.append({"image_path": path, "label": label, "animal": animal})

else:
    # a folder with one subfolder per label
    for label in os.listdir(INPUT):
        folder = os.path.join(INPUT, label)
        if os.path.isdir(folder):
            for file in os.listdir(folder):
                rows.append({"image_path": os.path.join(folder, file), "label": label})

data = pd.DataFrame(rows)
if "animal" not in data.columns:
    data["animal"] = data["image_path"]      # no animal ids: every image counts as its own animal
print("loaded:", len(data))


# ======================================================================================
# 2. Remove rows with a missing path or label
# ======================================================================================
data = data.dropna(subset=["image_path", "label"])
data["label"] = data["label"].astype(str).str.strip()
data = data[data["label"] != ""]


# ======================================================================================
# 3. One row per image. An image listed twice with different findings gets them joined: "fracture|tumor"
# ======================================================================================
def join_labels(labels):
    # ["fracture", "tumor|fracture"] -> "fracture|tumor"  (every finding once, in alphabetical order)
    findings = []
    for label in labels:
        for finding in label.split("|"):
            finding = finding.strip()
            if finding != "" and finding not in findings:
                findings.append(finding)
    return "|".join(sorted(findings))

data = data.groupby("image_path", as_index=False).agg({"label": join_labels, "animal": "first"})
data["original"] = data["image_path"]       # where the image came from (to find its COCO outlines later)
data["augmented"] = False


# ======================================================================================
# 4. Open every image: is it broken? how big is it? and its "fingerprint" (md5 hash)
# ======================================================================================
def check_image(path):
    # returns (md5 fingerprint, smallest side in pixels), or (None, 0) for a broken / missing file
    try:
        with Image.open(path) as img:
            img.verify()                          # fails on broken / cut-off files
        with Image.open(path) as img:
            size = min(img.size)
        with open(path, "rb") as f:
            md5 = hashlib.md5(f.read()).hexdigest()   # same content -> same fingerprint
        return md5, size
    except Exception:
        return None, 0

md5_list = []
size_list = []
for path in data["image_path"]:
    md5, size = check_image(path)
    md5_list.append(md5)
    size_list.append(size)
data["md5"] = md5_list
data["size"] = size_list

broken = data["md5"].isna()
thrown_out = data[broken].copy()
thrown_out["reason"] = "broken or missing"
rejected.append(thrown_out)
data = data[~broken]


# ======================================================================================
# 5. Remove duplicate images (same content saved twice).
#    If the copies have DIFFERENT labels we can't trust either one, so all of them go.
# ======================================================================================
labels_per_fingerprint = data.groupby("md5")["label"].transform("nunique")   # how many different labels share it
conflict = labels_per_fingerprint > 1
duplicate = data.duplicated("md5") & ~conflict                               # 2nd, 3rd ... copy of the same image

thrown_out = data[conflict].copy()
thrown_out["reason"] = "duplicate with different labels"
rejected.append(thrown_out)
thrown_out = data[duplicate].copy()
thrown_out["reason"] = "duplicate"
rejected.append(thrown_out)
data = data[~conflict & ~duplicate]


# ======================================================================================
# 6. Remove images that are too small to be a real X-ray
# ======================================================================================
too_small = data["size"] < MIN_SIZE
thrown_out = data[too_small].copy()
thrown_out["reason"] = "too small"
rejected.append(thrown_out)
data = data[~too_small]
print("after cleaning:", len(data))


# ======================================================================================
# 7. Shrink images that are too big (keeps the shape). The smaller copy is saved in OUTPUT/resized.
# ======================================================================================
new_paths = []
resized_count = 0
for i in range(len(data)):
    path = data["image_path"].iloc[i]
    with Image.open(path) as img:
        if max(img.size) > MAX_SIZE:
            img.thumbnail((MAX_SIZE, MAX_SIZE), Image.LANCZOS)    # shrink, keep the shape
            path = f"{OUTPUT}/resized/{data['md5'].iloc[i]}.png"
            img.save(path)
            resized_count = resized_count + 1
    new_paths.append(path)
print("resized:", resized_count)
data["image_path"] = new_paths


# ======================================================================================
# 8. Label names -> numbers, in alphabetical order: {"fracture": 0, "normal": 1}
# ======================================================================================
all_findings = []
for labels in data["label"]:
    for finding in labels.split("|"):
        if finding not in all_findings:
            all_findings.append(finding)
all_findings = sorted(all_findings)

label_map = {}
for number in range(len(all_findings)):
    label_map[all_findings[number]] = number

label_ids = []
for labels in data["label"]:
    numbers = []
    for finding in labels.split("|"):
        numbers.append(str(label_map[finding]))
    label_ids.append("|".join(numbers))        # e.g. "fracture|tumor" -> "0|5"
data["label_ids"] = label_ids


# ======================================================================================
# 9. Split into train / val / test (70 / 15 / 15) by ANIMAL, not by image:
#    all X-rays of one animal go to the same split, so test only has animals the model never saw.
#    Animals are grouped by their findings ("fracture", "normal", "fracture|normal"), and each group
#    is split on its own so every split gets the same mix. Train always keeps at least 1 animal per group.
# ======================================================================================
animals = data.groupby("animal", as_index=False)["label"].agg(join_labels)   # one row per animal: its findings

train_animals = []
val_animals = []
test_animals = []

# animals from other scanners: train only
from_other_source = animals["animal"].str.startswith(TRAIN_ONLY)
for animal in animals[from_other_source]["animal"]:
    train_animals.append(animal)
animals = animals[~from_other_source]

for findings, group in animals.groupby("label"):
    names = list(group["animal"].sample(frac=1, random_state=SEED))   # the animals of this group, shuffled
    n = len(names)
    n_test = min(round(n * TEST_SIZE), n - 1)
    n_val = min(round(n * VAL_SIZE), n - 1 - n_test)
    for k in range(n):
        if k < n_test:
            test_animals.append(names[k])
        elif k < n_test + n_val:
            val_animals.append(names[k])
        else:
            train_animals.append(names[k])

train = data[data["animal"].isin(train_animals)].copy()
val = data[data["animal"].isin(val_animals)].copy()
test = data[data["animal"].isin(test_animals)].copy()


# ======================================================================================
# 10. Copy the kept images into OUTPUT/train, valid, test (same layout as the input)
# ======================================================================================
for folder, part in [("train", train), ("valid", val), ("test", test)]:
    new_paths = []
    for path in part["image_path"]:
        new_path = f"{OUTPUT}/{folder}/{os.path.basename(path)}"
        shutil.copy(path, new_path)
        new_paths.append(new_path)
    part["image_path"] = new_paths


# ======================================================================================
# 11. Augmented copies of the TRAIN images only (val and test must stay real X-rays)
# ======================================================================================
torch.manual_seed(SEED)
transform = augment()
copies = []
for i in range(len(train)):
    row = train.iloc[i]
    img = Image.open(row["image_path"]).convert("RGB")
    for k in range(AUGMENT_COPIES):
        path = f"{OUTPUT}/train/aug_{row['md5']}_{k}.png"
        transform(img).save(path)                 # a slightly changed copy (flip, small turn, brightness ...)
        copy = row.copy()
        copy["image_path"] = path
        copy["augmented"] = True
        copies.append(copy)
train = pd.concat([train, pd.DataFrame(copies)])


# ======================================================================================
# 12. One _annotations.coco.json per folder, like the input.
#     Every image gets its "label". Real images keep their fracture outlines;
#     augmented copies get none (the outline no longer fits after turning / shifting).
# ======================================================================================
for folder, part in [("train", train), ("valid", val), ("test", test)]:
    coco = {"categories": categories, "images": [], "annotations": []}
    for i in range(len(part)):
        row = part.iloc[i]
        image_id = len(coco["images"])
        with Image.open(row["image_path"]) as img:
            width, height = img.size
        coco["images"].append({"id": image_id, "file_name": os.path.basename(row["image_path"]),
                               "width": width, "height": height, "label": row["label"],
                               "animal": row["animal"], "augmented": bool(row["augmented"])})
        if not row["augmented"]:
            for outline in outlines.get(row["original"], []):
                new_outline = dict(outline)                      # a copy of the outline ...
                new_outline["id"] = len(coco["annotations"])      # ... with new id numbers
                new_outline["image_id"] = image_id
                coco["annotations"].append(new_outline)
    with open(f"{OUTPUT}/{folder}/_annotations.coco.json", "w") as f:
        json.dump(coco, f)


# ======================================================================================
# 13. Save everything
# ======================================================================================
train.to_csv(OUTPUT + "/train.csv", index=False)
val.to_csv(OUTPUT + "/val.csv", index=False)
test.to_csv(OUTPUT + "/test.csv", index=False)
all_rejected = pd.concat(rejected)
all_rejected.to_csv(OUTPUT + "/rejected.csv", index=False)
with open(OUTPUT + "/label_map.json", "w") as f:
    json.dump(label_map, f, indent=2)

print("labels:", label_map)
print("train:", len(train), "val:", len(val), "test:", len(test), "rejected:", len(all_rejected))
