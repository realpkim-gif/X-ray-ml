from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import pandas as pd
import torch
import numpy as np

processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
model = AutoModel.from_pretrained('facebook/dinov2-base')


for split in ["train", "val", "test"]:
    input = pd.read_csv(f"data/processed/{split}.csv")["image_path"]  # image paths from preprocess.py
    output = []

    with torch.no_grad():  # no gradients needed, model is frozen
        for i in input:
            inputs = processor(images=i, return_tensors="pt")
            outputs = model(**inputs)
            last_hidden_states = outputs.last_hidden_state
            img = Image.open(i).convert("RGB")  # gray X-ray -> 3 channels
            output.append(last_hidden_states = outputs.last_hidden_state)  # unsqueeze single image into batch of 1

    output = torch.cat(output).numpy()  # [num_images, 1024]
    np.save(f"data/features/dino_{split}.npy", output)  # row i = image i of {split}.csv
