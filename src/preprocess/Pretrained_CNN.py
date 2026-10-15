import os
from PIL import Image
import timm
import torch
import numpy as np
import pandas as pd
# ---------- always work from the project folder (X-ray-ml) ----------
# This file lives in src/<folder>/, but all paths below ("data/...", "src/models/weights/...") start at the
# project folder. These lines go there first, so it works from PyCharm's Run button AND from a terminal.
PROJECT_FOLDER = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # up 3: file -> src/<folder> -> src -> project
os.chdir(PROJECT_FOLDER)


model = timm.create_model(
    'convnextv2_base.fcmae_ft_in22k_in1k_384',
    pretrained=True,
    num_classes=0,  # remove classifier -> outputs one 1024 feature vector per image
)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.eval().to(device)  # frozen: inference mode, on the gpu if there is one
for p in model.parameters():
    p.requires_grad = False  # frozen: weights never update

# get model specific transforms (normalization, resize)
data_config = timm.data.resolve_model_data_config(model)
# ** = "use every setting in data_config (size, mean, std, ...) as an argument"
transforms = timm.data.create_transform(**data_config, is_training=False)

os.makedirs("data/features", exist_ok=True)  # create the output folder if missing

for split in ["train", "val", "test"]:
    input = pd.read_csv(f"data/PreprocessedData/{split}.csv")["image_path"]  # image paths from preprocess.py
    output = []

    with torch.no_grad():  # no gradients needed, model is frozen
        for i in input:
            img = Image.open(i).convert("RGB")  # gray X-ray -> 3 channels
            output.append(model(transforms(img).unsqueeze(0).to(device)).cpu())  # unsqueeze single image into batch of 1

    output = torch.cat(output).numpy()  # [num_images, 1024]
    np.save(f"data/features/cnn_{split}.npy", output)  # row i = image i of {split}.csv
