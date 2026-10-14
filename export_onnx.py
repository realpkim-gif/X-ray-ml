import os
import json
import numpy as np
import pandas as pd
import torch
from torch import nn
import timm
from PIL import Image
from transformers import AutoModel
import onnx
from onnx import numpy_helper, helper
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
from sklearn.metrics import roc_auc_score
import main                      # CNN+DINO head (head.pt), thresholds
import FractureLocator as fl     # patch head (fracture_head.pt), find_box, box_iou

# Turns the whole model into ONNX files a web BROWSER can run (with onnxruntime-web), so no server is needed.
#
# ONNX = a standard file format for neural networks. PyTorch writes it, and "onnxruntime" runs it
# in Python, C++, phones... and in the browser (JavaScript).
#
# Three files go into web_model/:
#   cnn.onnx    ConvNeXt  : picture [1, 3, 384, 384] (0-1)  -> cnn vector [1, 1024]
#   dino.onnx   DINOv2    : picture [1, 3, 448, 448] (0-1)  -> dino vector [1, 1536] + patches [1, 32, 32, 768]
#   heads.onnx  both heads: vectors + dropout masks         -> CNN+DINO probability, patch-head probability, heatmap
# plus config.json (thresholds and settings the JavaScript needs).
#
# The two backbones are big (~350 MB each), so they are made smaller (see step 4).
# Step 5 checks that the browser version still gives the same answers as PyTorch.
#
#   python export_onnx.py

OUT = "web_model"
PASSES = 5                # Monte Carlo dropout passes per head (same as vote.py / API.py)
DROPOUT_HEAD = 0.3        # main.py Head: nn.Dropout(0.3)
DROPOUT_PATCH = 0.2       # FractureLocator PatchHead: nn.Dropout2d(0.2)
CHECK_SPLITS = ["val", "test"]
device = "cpu"            # export on the cpu (the browser runs on the cpu too)
os.makedirs(OUT, exist_ok=True)

MEAN = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)   # ImageNet mean/std: both backbones use them
STD = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)


# ======================================================================================
# 1. Small wrapper models: exactly what each ONNX file should do, start to finish
# ======================================================================================

class CnnNet(nn.Module):
    # 0-1 picture -> normalise -> ConvNeXt -> [1, 1024]   (normalising inside = less JavaScript)
    def __init__(self):
        super(CnnNet, self).__init__()
        self.model = timm.create_model("convnextv2_base.fcmae_ft_in22k_in1k_384", pretrained=True, num_classes=0)

    def forward(self, pixels):
        return self.model((pixels - MEAN) / STD)


class DinoNet(nn.Module):
    # 0-1 picture -> normalise -> DINOv2 -> the same two things DINO.py and DINO_patches.py saved
    def __init__(self):
        super(DinoNet, self).__init__()
        self.model = AutoModel.from_pretrained("facebook/dinov2-base")

    def forward(self, pixels):
        tokens = self.model(pixel_values=(pixels - MEAN) / STD).last_hidden_state   # [1, 1 + 1024, 768]
        dino = torch.cat([tokens[:, 0], tokens[:, 1:].mean(dim=1)], dim=1)          # summary + average [1, 1536]
        patches = tokens[:, 1:].reshape(-1, 32, 32, 768)                            # every square [1, 32, 32, 768]
        return dino, patches


class HeadsNet(nn.Module):
    # Both trained heads in one file, with Monte Carlo dropout done by MASKS:
    #   an ONNX file can't switch dropout on, so the browser makes the random masks itself
    #   (1 = keep this neuron, 0 = drop it) and passes them in. Same maths as nn.Dropout in train mode:
    #   multiply by the mask, then divide by the keep chance so the average stays the same.
    # Batch = PASSES: row i of every input is one Monte Carlo pass.
    def __init__(self, head, patch_head):
        super(HeadsNet, self).__init__()
        self.head = head
        self.patch_head = patch_head
        self.fracture = main.label_map["fracture"]

    def forward(self, cnn, dino, patches, keep_head, keep_patch):
        # CNN+DINO head (main.py Head.forward, with our own dropout)
        h = self.head
        x = h.relu(h.fc1(h.norm1(cnn)) + h.fc2(h.norm2(dino)))      # [B, 512]
        x = x * keep_head / (1 - DROPOUT_HEAD)                       # keep_head [B, 512]
        head_prob = torch.sigmoid(h.fc3(x)[:, self.fracture])        # [B] fracture probability

        # patch head (FractureLocator PatchHead.forward, with our own dropout)
        p = self.patch_head
        x = p.norm(patches).permute(0, 3, 1, 2)                       # [B, 768, 32, 32]
        x = p.relu(p.conv2(p.relu(p.conv1(x))))                       # [B, 256, 32, 32]
        x = x * keep_patch / (1 - DROPOUT_PATCH)                      # keep_patch [B, 256, 1, 1]: whole channels
        squares = p.conv3(x)[:, 0]                                    # [B, 32, 32] raw square scores
        # image probability = mean of the TOP_K highest squares, like fl.image_scores
        # (written with flatten here: fl.image_scores uses len(), which ONNX would freeze to exactly 5 images)
        top = torch.topk(squares.flatten(1), fl.TOP_K, dim=1).values  # [B, TOP_K]
        patch_prob = torch.sigmoid(top.mean(dim=1))                   # [B]
        heatmap = torch.sigmoid(squares)                              # [B, 32, 32] 0-1
        return head_prob, patch_prob, heatmap


# ======================================================================================
# 2. Load the trained heads and build the three wrappers
# ======================================================================================

head = main.Head(1024, 1536, main.num_classes)
head.load_state_dict(torch.load("head.pt", map_location=device))
patch_head = fl.PatchHead(768)
patch_head.load_state_dict(torch.load("fracture_head.pt", map_location=device))

cnn_net = CnnNet().eval()
dino_net = DinoNet().eval()
heads_net = HeadsNet(head, patch_head).eval()   # eval: the real nn.Dropout layers stay OFF; our masks do the dropout


# ======================================================================================
# 3. Export to ONNX. torch.onnx.export runs the model once on example inputs and records every step.
# ======================================================================================

def export(model, inputs, names, outputs, path, batch_axis=False):
    # dynamic_axes: which sizes may change later. For heads.onnx the batch (number of passes) may change.
    axes = {}
    if batch_axis:
        for name in names + outputs:
            axes[name] = {0: "batch"}
    torch.onnx.export(model, inputs, path, input_names=names, output_names=outputs,
                      dynamic_axes=axes, opset_version=17, dynamo=False)
    print("exported", path, round(os.path.getsize(path) / 1e6, 1), "MB")


with torch.no_grad():
    export(cnn_net, (torch.rand(1, 3, 384, 384),), ["pixels"], ["cnn"], f"{OUT}/cnn_fp32.onnx")
    export(dino_net, (torch.rand(1, 3, 448, 448),), ["pixels"], ["dino", "patches"], f"{OUT}/dino_fp32.onnx")
    example = (torch.rand(PASSES, 1024), torch.rand(PASSES, 1536), torch.rand(PASSES, 32, 32, 768),
               torch.ones(PASSES, 512), torch.ones(PASSES, 256, 1, 1))
    export(heads_net, example, ["cnn", "dino", "patches", "keep_head", "keep_patch"],
           ["head_prob", "patch_prob", "heatmap"], f"{OUT}/heads.onnx", batch_axis=True)


# ======================================================================================
# 4. Make the two backbones smaller, in the way that keeps each one accurate:
#    - ConvNeXt: weights as 8-bit integers (~4x smaller). Checked: the CNN+DINO head gives the same answers.
#    - DINO: 8-bit was NOT good enough (the patch head needs very fine detail per square: its fracture
#      probability fell from 1.0 to 0.03). So DINO's weights are only STORED as 16-bit numbers (2x smaller)
#      and turned back into normal 32-bit numbers when the model loads, so all the maths stays exact.
# ======================================================================================

def store_weights_16bit(path_in, path_out):
    # every big weight table (initializer) -> 16-bit copy + a "Cast" step that turns it back into 32-bit
    model = onnx.load(path_in)
    graph = model.graph
    casts = []
    for weight in graph.initializer:
        if weight.data_type == onnx.TensorProto.FLOAT and np.prod(weight.dims) > 1000:   # skip tiny ones
            values = numpy_helper.to_array(weight).astype(np.float16)
            original_name = weight.name
            weight.CopyFrom(numpy_helper.from_array(values, original_name + "_16bit"))
            # the Cast node produces the old name again, so the rest of the network doesn't notice
            casts.append(helper.make_node("Cast", [original_name + "_16bit"], [original_name], to=onnx.TensorProto.FLOAT))
    for node in reversed(casts):
        graph.node.insert(0, node)                         # casts first, before anything uses the weights
    onnx.save(model, path_out)


quantize_dynamic(f"{OUT}/cnn_fp32.onnx", f"{OUT}/cnn.onnx",
                 weight_type=QuantType.QUInt8, op_types_to_quantize=["MatMul", "Gemm"])
store_weights_16bit(f"{OUT}/dino_fp32.onnx", f"{OUT}/dino.onnx")
for name in ["cnn", "dino"]:
    os.remove(f"{OUT}/{name}_fp32.onnx")
    print("smaller", f"{OUT}/{name}.onnx", round(os.path.getsize(f"{OUT}/{name}.onnx") / 1e6, 1), "MB")

# everything the JavaScript needs besides the models
with open("thresholds.json") as f:
    head_threshold = json.load(f)["fracture"]
config = {
    "head_threshold": head_threshold,                                     # CNN+DINO head votes "fracture" above this
    "patch_threshold": float(pd.read_csv("fracture_metrics.csv")["threshold"][0]),   # patch head votes above this
    "passes": PASSES,
    "keep_head": 1 - DROPOUT_HEAD,
    "keep_patch": 1 - DROPOUT_PATCH,
    "box_threshold": fl.BOX_THRESHOLD,
    "grid": 32,
}
with open(f"{OUT}/config.json", "w") as f:
    json.dump(config, f, indent=2)
print("config:", config)


# ======================================================================================
# 5. CHECK: does the browser version give the same answers as PyTorch?
#    The browser resizes pictures with a <canvas> (smooth, bilinear-like), PyTorch training used bicubic,
#    and the backbones are now smaller. So we run the ONNX files the way the browser will, without dropout
#    (all masks = 1, so the answers are not random), and compare with the saved PyTorch features.
# ======================================================================================

cnn_session = ort.InferenceSession(f"{OUT}/cnn.onnx")
dino_session = ort.InferenceSession(f"{OUT}/dino.onnx")
heads_session = ort.InferenceSession(f"{OUT}/heads.onnx")


def like_browser(path, size):
    # what radiant.js does: stretch to 1024 x 1024, then smooth-resize to the backbone's size, 0-1 numbers
    img = Image.open(path).convert("RGB").resize((1024, 1024), Image.BILINEAR).resize((size, size), Image.BILINEAR)
    return (np.asarray(img, dtype=np.float32) / 255).transpose(2, 0, 1)[None]   # [1, 3, size, size]


def run_heads(cnn, dino, patches):
    n = len(cnn)
    return heads_session.run(None, {"cnn": cnn, "dino": dino, "patches": patches.astype(np.float32),
                                    "keep_head": np.ones((n, 512), np.float32),
                                    "keep_patch": np.ones((n, 256, 1, 1), np.float32)})


for split in CHECK_SPLITS:
    data, saved_patches, masks, labels = fl.load(split, keep_flipped=False)
    saved_cnn, saved_dino, _ = main.load(split)
    y = labels.numpy()

    # A. PyTorch features (saved .npy, as in training)
    torch_head, torch_patch, torch_heat = run_heads(saved_cnn.cpu().numpy(), saved_dino.cpu().numpy(), saved_patches.numpy())

    # B. browser-style: canvas resize + 8-bit backbones
    web_cnn, web_dino, web_patches = [], [], []
    for path in data["image_path"]:
        web_cnn.append(cnn_session.run(None, {"pixels": like_browser(path, 384)})[0])
        dino, patches = dino_session.run(None, {"pixels": like_browser(path, 448)})
        web_dino.append(dino)
        web_patches.append(patches)
    web_head, web_patch, web_heat = run_heads(np.concatenate(web_cnn), np.concatenate(web_dino), np.concatenate(web_patches))

    # compare: AUROC of each, and how often the yes/no vote and the box agree
    agree_head = ((torch_head >= config["head_threshold"]) == (web_head >= config["head_threshold"])).mean()
    agree_patch = ((torch_patch >= config["patch_threshold"]) == (web_patch >= config["patch_threshold"])).mean()
    ious = []
    for i in np.where(y == 1)[0]:
        ious.append(fl.box_iou(fl.find_box(torch_heat[i]), fl.find_box(web_heat[i])))
    print(f"\n{split}: AUROC CNN+DINO torch {roc_auc_score(y, torch_head):.3f} / browser {roc_auc_score(y, web_head):.3f}"
          f" | patch head torch {roc_auc_score(y, torch_patch):.3f} / browser {roc_auc_score(y, web_patch):.3f}")
    print(f"{split}: same vote CNN+DINO {agree_head:.1%}, patch head {agree_patch:.1%}, "
          f"box overlap torch vs browser (fractures) {np.mean(ious):.2f}")
