import os
import json
from transformers import AutoImageProcessor, AutoModel
from PIL import Image, ImageDraw, ImageOps
import pandas as pd
import torch
import numpy as np

# Like DINO.py, but keeps EVERY patch vector instead of averaging them.
# That way the head can learn WHERE the fracture is, not just IF there is one.
#
#   448 x 448 image, cut into 14 x 14 pixel patches = 32 x 32 squares
#   every square gets a 768 number vector from DINO  ->  [32, 32, 768] per image
#
# It also turns the fracture outlines into the same 32 x 32 grid (the "mask"):
#   0 = healthy square, 1 = square fully inside the fracture, 0.5 = half inside
#
# Files written for each split (train / val / test):
#   data/features/patches_{split}.npy   [N, 32, 32, 768]
#   data/features/masks_{split}.npy     [N, 32, 32]
#   data/features/patches_{split}.csv   row i = image i of the two .npy files

DATA = "PreprocessedData"   # where preprocess.py writes the csvs
SIZE = 448                  # image size for DINO, must be a multiple of 14
GRID = SIZE // 14           # 32 squares per side
FLIP = True                 # also save a mirrored copy of every image (more training examples).
                            # FractureLocator only uses the mirrored copies for TRAINING.

device = "cuda" if torch.cuda.is_available() else "cpu"

# resize the whole image to SIZE x SIZE, no center crop (a crop could cut off a fracture near the edge)
processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base',
                                               size={"height": SIZE, "width": SIZE}, do_center_crop=False)
model = AutoModel.from_pretrained('facebook/dinov2-base').to(device).eval()   # frozen: inference mode

os.makedirs("data/features", exist_ok=True)  # create the output folder if missing


def read_outlines(json_path):
    # Reads the COCO json and returns a dictionary:
    #   file name -> list of outlines, each outline is a list of (x, y) points
    with open(json_path) as f:
        coco = json.load(f)

    # the json links outlines to images by id, so first make: id -> file name
    file_names = {}
    for image in coco["images"]:
        file_names[image["id"]] = image["file_name"]

    outlines = {}
    for annotation in coco["annotations"]:
        file_name = file_names[annotation["image_id"]]
        if file_name not in outlines:
            outlines[file_name] = []

        # COCO stores an outline as one flat list: [x1, y1, x2, y2, x3, y3, ...]
        for flat_points in annotation["segmentation"]:
            points = []
            for i in range(0, len(flat_points), 2):
                points.append((flat_points[i], flat_points[i + 1]))
            outlines[file_name].append(points)
    return outlines


def make_mask(image_outlines, width, height):
    # Draws the outlines as white shapes on a black image the same size as the X-ray.
    # Black (0) = healthy, white (255) = fracture.
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for points in image_outlines:
        draw.polygon(points, fill=255)
    return mask


def mask_to_grid(mask):
    # Shrinks the mask to 32 x 32. Image.BOX averages all pixels in a square,
    # so each square becomes how much of it was fracture (0 to 1).
    small = mask.resize((GRID, GRID), Image.BOX)
    return np.array(small, dtype=np.float32) / 255


def get_patches(img):
    # Runs DINO on one image and returns its 32 x 32 grid of patch vectors.
    inputs = processor(images=img, return_tensors="pt").to(device)
    tokens = model(**inputs).last_hidden_state   # [1, 1 + 1024, 768]: 1 summary token + 1024 patches
    tokens = tokens[0, 1:]                       # drop the summary token -> [1024, 768]
    grid = tokens.reshape(GRID, GRID, -1)        # [32, 32, 768]
    return grid.half().cpu()                     # half precision = half the disk space


# the csv files are called train / val / test, but the image folders are train / valid / test
splits = {"train": "train", "val": "valid", "test": "test"}

for split in splits:
    folder = splits[split]
    data = pd.read_csv(f"{DATA}/{split}.csv")
    data = data[data["augmented"] == False]     # augmented copies have no outlines -> skip them
    outlines = read_outlines(f"{DATA}/{folder}/_annotations.coco.json")

    if FLIP:
        versions = [False, True]    # first all real images, then all mirrored images
    else:
        versions = [False]

    rows = []       # one row of info per image, becomes the csv
    patches = []    # one [32, 32, 768] grid per image
    masks = []      # one [32, 32] mask per image

    with torch.no_grad():  # no gradients needed, model is frozen
        for flip in versions:
            for _, row in data.iterrows():
                img = Image.open(row["image_path"]).convert("RGB")  # gray X-ray -> 3 channels

                file_name = os.path.basename(row["image_path"])
                image_outlines = outlines.get(file_name, [])        # no outlines = no fracture
                mask = make_mask(image_outlines, img.width, img.height)

                if flip:                                 # mirror the image AND its mask, so they still match
                    img = ImageOps.mirror(img)
                    mask = ImageOps.mirror(mask)

                patches.append(get_patches(img))
                masks.append(mask_to_grid(mask))
                rows.append([row["image_path"], row["original"], row["animal"], row["label"], flip])

    # save everything for this split
    patches = torch.stack(patches).numpy()      # [N, 32, 32, 768]
    masks = np.stack(masks)                     # [N, 32, 32]
    np.save(f"data/features/patches_{split}.npy", patches)
    np.save(f"data/features/masks_{split}.npy", masks)

    info = pd.DataFrame(rows, columns=["image_path", "original", "animal", "label", "flipped"])
    info.to_csv(f"data/features/patches_{split}.csv", index=False)

    fracture_count = (info["label"] == "fracture").sum()
    print(split, patches.shape, "fracture images:", fracture_count)
