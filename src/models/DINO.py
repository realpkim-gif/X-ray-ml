import os
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import pandas as pd
import torch
import numpy as np
# ---------- always work from the project folder (X-ray-ml) ----------
# This file lives in src/<folder>/, but all paths below ("data/...", "src/models/weights/...") start at the
# project folder. These lines go there first, so it works from PyCharm's Run button AND from a terminal.
PROJECT_FOLDER = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # up 3: file -> src/<folder> -> src -> project
os.chdir(PROJECT_FOLDER)


DATA = "data/PreprocessedData"   # where preprocess.py writes the csvs
SIZE = 448                  # bigger than the usual 224 so small fractures don't vanish (must be a multiple of 14)

device = "cuda" if torch.cuda.is_available() else "cpu"

# resize the whole image to SIZE x SIZE, no center crop (a crop could cut off a fracture near the edge)
processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base',
                                               size={"height": SIZE, "width": SIZE}, do_center_crop=False)
model = AutoModel.from_pretrained('facebook/dinov2-base').to(device).eval()   # frozen: inference mode

os.makedirs("data/features", exist_ok=True)  # create the output folder if missing

for split in ["train", "val", "test"]:
    input = pd.read_csv(f"{DATA}/{split}.csv")["image_path"]  # image paths from preprocess.py
    output = []

    with torch.no_grad():  # no gradients needed, model is frozen
        for i in input:
            img = Image.open(i).convert("RGB")  # gray X-ray -> 3 channels
            inputs = processor(images=img, return_tensors="pt").to(device)
            tokens = model(pixel_values=inputs["pixel_values"]).last_hidden_state    # [1, 1 + num_patches, 768]
            cls = tokens[:, 0]                  # summary token of the whole image   [1, 768]
            patches = tokens[:, 1:].mean(dim=1) # average of all small patches       [1, 768]
            output.append(torch.cat([cls, patches], dim=1).cpu())   # [1, 1536]

    output = torch.cat(output).numpy()  # [num_images, 1536]
    np.save(f"data/features/dino_{split}.npy", output)  # row i = image i of {split}.csv
    print(split, output.shape)
