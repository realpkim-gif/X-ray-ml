import os
import json
import numpy as np
import torch
from torch import nn
import onnx
from onnx import numpy_helper, helper
import onnxruntime as ort
from PIL import Image

# ---------- always work from the project folder (X-ray-ml) ----------
# This file lives in src/<folder>/, but all paths below ("data/...", "src/models/weights/...") start at the
# project folder. These lines go there first, so it works from PyCharm's Run button AND from a terminal.
PROJECT_FOLDER = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # up 3: file -> src/<folder> -> src -> project
os.chdir(PROJECT_FOLDER)

from Xray_Or_Not import resnet, load_head, xray_chance, device   # the frozen ResNet18 and the trained gate head

# ======================================================================================
# export_gate_onnx.py - the "X-ray or not" gate as ONE file the browser can run: website/web_model/gate.onnx
#
#   picture [1, 3, 224, 224] (numbers 0-1)  ->  normalise  ->  ResNet18  ->  gate head  ->  X-ray chance (0-1)
#
# radiant.js runs this FIRST. If the chance is below GATE_THRESHOLD, the website asks
# "We believe this isn't an X-ray. Still run?" instead of running the big fracture model.
#
#   python src/models/export_gate_onnx.py
# ======================================================================================

OUT = "src/website/web_model"
GATE_THRESHOLD = 0.5          # X-ray chance below this = "not an X-ray" (same as Xray_Or_Not.py)
MEAN = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)   # ImageNet stats, what ResNet18 was trained with
STD = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)


class GateNet(nn.Module):
    # everything the gate does, start to finish, in one model
    def __init__(self, resnet, head):
        super(GateNet, self).__init__()
        self.resnet = resnet
        self.head = head

    def forward(self, pixels):
        x = (pixels - MEAN) / STD                    # normalise inside the model -> less JavaScript
        x = self.resnet(x)                           # [1, 512]
        scores = self.head(x)                        # [1, 2]  ("not X-ray", "X-ray")
        return torch.softmax(scores, dim=1)[:, 1]    # [1]  chance of "X-ray"


def store_weights_16bit(path_in, path_out):
    # same trick as website/export_onnx.py: weights STORED as 16-bit (half the file size),
    # turned back into 32-bit when the model loads, so the maths stays exact
    model = onnx.load(path_in)
    casts = []
    for weight in model.graph.initializer:
        if weight.data_type == onnx.TensorProto.FLOAT and np.prod(weight.dims) > 1000:
            values = numpy_helper.to_array(weight).astype(np.float16)
            name = weight.name
            weight.CopyFrom(numpy_helper.from_array(values, name + "_16bit"))
            casts.append(helper.make_node("Cast", [name + "_16bit"], [name], to=onnx.TensorProto.FLOAT))
    for node in reversed(casts):
        model.graph.node.insert(0, node)
    onnx.save(model, path_out)


# ---------- 1. export ----------
gate = GateNet(resnet.cpu(), load_head().cpu()).eval()
with torch.no_grad():
    torch.onnx.export(gate, (torch.rand(1, 3, 224, 224),), f"{OUT}/gate_fp32.onnx",
                      input_names=["pixels"], output_names=["xray_chance"], opset_version=17, dynamo=False)
store_weights_16bit(f"{OUT}/gate_fp32.onnx", f"{OUT}/gate.onnx")
os.remove(f"{OUT}/gate_fp32.onnx")
print("saved", f"{OUT}/gate.onnx", round(os.path.getsize(f"{OUT}/gate.onnx") / 1e6, 1), "MB")

# ---------- 2. add the gate threshold to config.json (radiant.js reads it) ----------
with open(f"{OUT}/config.json") as f:
    config = json.load(f)
config["gate_threshold"] = GATE_THRESHOLD
with open(f"{OUT}/config.json", "w") as f:
    json.dump(config, f, indent=2)

# ---------- 3. check: does the ONNX gate give the same chance as PyTorch? ----------
# (pictures resized the way the browser does: straight to 224 x 224 with a canvas, like PIL BILINEAR)
session = ort.InferenceSession(f"{OUT}/gate.onnx")
resnet.to(device)                                # back to the GPU (it was moved to the cpu for the export)
head = load_head()
biggest_difference = 0
for label in ["1", "0"]:
    folder = f"data/challenge_web/{label}"
    for file in sorted(os.listdir(folder))[:10]:
        picture = Image.open(f"{folder}/{file}").convert("RGB")
        small = picture.resize((224, 224), Image.BILINEAR)
        pixels = (np.asarray(small, dtype=np.float32) / 255).transpose(2, 0, 1)[None]   # [1, 3, 224, 224]
        onnx_chance = float(session.run(None, {"pixels": pixels})[0][0])
        torch_chance = xray_chance(head, picture)
        biggest_difference = max(biggest_difference, abs(onnx_chance - torch_chance))
print("biggest difference ONNX vs PyTorch on 20 web pictures:", round(biggest_difference, 4))
