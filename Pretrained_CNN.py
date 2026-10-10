import os
from PIL import Image
import timm
import torch
import numpy as np
import pandas as pd

model = timm.create_model(
    'convnextv2_base.fcmae_ft_in22k_in1k_384',
    pretrained=True,
    num_classes=0,  # remove classifier -> outputs one 1024 feature vector per image
)
model = model.eval()  # frozen: inference mode
for p in model.parameters():
    p.requires_grad = False  # frozen: weights never update

# get model specific transforms (normalization, resize)
data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)

os.makedirs("data/features", exist_ok=True)  # create the output folder if missing

for split in ["train", "val", "test"]:
    input = pd.read_csv(f"data/processed/{split}.csv")["image_path"]  # image paths from preprocess.py
    output = []

    with torch.no_grad():  # no gradients needed, model is frozen
        for i in input:
            img = Image.open(i).convert("RGB")  # gray X-ray -> 3 channels
            output.append(model(transforms(img).unsqueeze(0)))  # unsqueeze single image into batch of 1

    output = torch.cat(output).numpy()  # [num_images, 1024]
    np.save(f"data/features/cnn_{split}.npy", output)  # row i = image i of {split}.csv
