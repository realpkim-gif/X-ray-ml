# Running the fracture model in the browser (no server)

## The idea

A **static** website (GitHub Pages, or a "Static" Hugging Face Space) can only send files to the visitor.
It can't run Python. So instead of running the model on a server, we send the **model itself** to the visitor's
browser, and the browser runs it with **onnxruntime-web**, a JavaScript library that runs neural networks.

```
Visitor's browser
  1. downloads the model files once (~280 MB, then the browser caches them)
  2. user picks an X-ray -> radiant.js runs ConvNeXt + DINO + both heads ON THE VISITOR'S COMPUTER
  3. shows the answer + the X-ray with box / heatmap / both
```

Good side: free forever, no server to keep running, and **X-rays never leave the visitor's computer** (private).
Down side: the first visit downloads ~280 MB, and each X-ray takes ~10-25 seconds on a normal laptop.

---

## The files

| File | Made by | What it is |
|---|---|---|
| `cnn.onnx` (100 MB) | export_onnx.py | ConvNeXt: picture 384x384 -> cnn vector [1, 1024]. Weights stored as 8-bit numbers |
| `dino.onnx` (174 MB) | export_onnx.py | DINOv2: picture 448x448 -> dino vector [1, 1536] + patches [1, 32, 32, 768]. Weights stored as 16-bit numbers |
| `gate.onnx` (23 MB) | src/models/export_gate_onnx.py | the "X-ray or not" gate (ResNet18 + head): picture 224x224 -> X-ray chance. Runs first |
| `heads.onnx` (8 MB) | export_onnx.py | both trained heads, 5 Monte Carlo passes at once -> votes + heatmaps |
| `config.json` | export_onnx.py | the two vote thresholds, number of passes, dropout keep chances, box setting |

These plain files stay in `src/models/weights/onnx/` and are **never uploaded**. `src/website/encrypt_models.py`
locks them and writes `part0-0.bin` (config) ... `part4-0.bin` (heads), in 45 MB pieces, into the website repo's `model/` folder.
`radiant.js` and the page that uses it (`Analyze.dc.html`) live in the **X-ray.web** repo.

**ONNX** is a standard file format for neural networks: PyTorch writes it, and onnxruntime runs it in Python,
on phones, or in the browser.

---

## How `export_onnx.py` works (step by step)

1. **Wrapper models.** Small PyTorch classes that do exactly what each ONNX file should do, start to finish
   (including the normalisation, so the JavaScript only has to divide pixels by 255).
2. **Monte Carlo dropout without dropout.** An ONNX file can't switch dropout on. So `HeadsNet` takes **masks**
   as inputs (1 = keep a neuron, 0 = drop it). The browser makes random masks each time:
   `x = x * mask / keep_chance`, which is exactly what `nn.Dropout` does in train mode.
   The batch is 5 rows = 5 passes, all run in one go.
3. **Export** with `torch.onnx.export`: it runs the model once on example inputs and records every step.
4. **Make the backbones smaller**, each in the way that keeps it accurate:
   - **ConvNeXt:** 8-bit integers (4x smaller). The CNN+DINO head gave the same answers.
   - **DINO:** 8-bit was **not** good enough. The patch head needs very fine detail in every square, and its fracture
     probability fell from **1.0 to 0.03**. So DINO's weights are only **stored** as 16-bit (2x smaller) and turned
     back into 32-bit when the model loads, so the maths is exact.
5. **Check** against PyTorch on all val/test X-rays, resizing pictures the way the browser does.

Run it again whenever you retrain a head, then lock the new files:
```bash
python src/website/export_onnx.py
python src/website/encrypt_models.py
```

---

## Encryption (`encrypt_models.py`)

The browser has to download the model to run it, so plain `.onnx` files could just be saved and reused.
So every file is encrypted with **AES-256-GCM** (a strong, standard encryption; GCM also detects changed files):

1. The key: 32 random bytes, made once and kept in `src/models/weights/model_key.txt` (in `.gitignore`).
2. Each file: 12 new random bytes (the IV) + the encrypted bytes -> cut into 45 MB pieces `part0-0.bin` ... `part4-0.bin`.
3. The key is written into `radiant.js` as two scrambled halves (`KEY_A` xor `KEY_B`).
4. In the browser, `radiant.js` downloads the pieces of each file, glues them together, unlocks it in memory with `crypto.subtle.decrypt`
   (built into every browser), and hands the bytes straight to onnxruntime. The unlocked model is never saved.

**Honest limit:** the browser needs the key to unlock the files, so the key is in the public JavaScript.
Someone skilled could find it and decrypt the files. This stops casual "save the file" copying; the license in
README section 14 is what makes copying not allowed.

---

## How `radiant.js` works

`Radiant.load(folder)` downloads and unlocks the 4 models + config once.
`Radiant.analyse(file, show)` does, for one X-ray:

1. Draw the X-ray on a 1024x1024 `<canvas>` (stretched, like the training X-rays).
2. Shrink it to 384x384 -> `cnn.onnx`, and to 448x448 -> `dino.onnx`.
   `toTensor()` turns canvas pixels into numbers 0-1 in the order `[1, 3 colours, height, width]`.
3. Copy the vectors 5 times (one per pass), make random dropout masks, run `heads.onnx` once.
4. Count votes: each pass of each head votes "fracture" if its probability is above its threshold.
   10 votes -> majority = answer (5-5 counts as fracture), share agreeing = confidence.
5. Average the 5 heatmaps; if fracture, `findBox()` grows a box from the hottest square
   (same rule as `FractureLocator.find_box`).
6. `Radiant.redraw(result, show)` paints the heatmap and/or box on the **original** picture and gives a PNG data URL.

The result looks like this:
```javascript
{ answer: "fracture", confidence: 1, fracture_votes: "10 of 10",
  box: {x1: 0.5313, y1: 0.3438, x2: 0.875, y2: 0.625},   // 0-1 fractions, null for "normal"
  heatmap: [[...32 numbers...], ...32 rows],
  show: "both", image: "data:image/png;base64,...", disclaimer: "Not a diagnosis..." }
```
`Radiant.redraw(result, "box")` switches the picture **instantly**, without running the model again.

---

## Try it on your laptop

From the **X-ray.web** folder:
```bash
python -m http.server 8090
```
Then open http://localhost:8090/Analyze.dc.html. On localhost the page loads the models from `./model/`.
(Double-clicking the page won't work: browsers block loading the model files from `file://`.)

**Tested in the browser** (encrypted files, Analyze page): a fracture X-ray gave 10 of 10 votes and the same box as
PyTorch in ~12 s; a photo was stopped by the gate in 0.1 s.

**Checked on all val/test X-rays** (export_onnx.py step 5, browser-style resizing, no dropout):

| | AUROC PyTorch -> browser | same yes/no vote | box overlap browser vs PyTorch |
|---|---|---|---|
| val, CNN+DINO head | 0.928 -> 0.916 | 96.8% | - |
| val, patch head | 0.995 -> 0.995 | 99.5% | 0.96 |
| test, CNN+DINO head | 0.975 -> 0.981 | 95.9% | - |
| test, patch head | 0.942 -> 0.947 | 99.4% | 0.92 |

---

## Putting it online

Everything is in the X-ray.web repo on GitHub Pages: the page, `radiant.js`, and the encrypted model in `model/`.
GitHub refuses single files over 100 MB, so `encrypt_models.py` (Python) cuts every encrypted file into pieces of
45 MB or less: `part3.bin` (DINO, 174 MB) becomes `part3-0.bin` ... `part3-3.bin`, 10 pieces in total. It writes how
many pieces each file has into `radiant.js` (`PIECES = [1, 1, 3, 4, 1]`).
In the browser, `fetchAndDecrypt` in `radiant.js` (JavaScript) downloads the pieces, glues them back together
with `new Blob(pieces)`, then decrypts. See the X-ray.web README for the code.

**After retraining:** `export_onnx.py` (and/or `export_gate_onnx.py`), then `encrypt_models.py`, then commit and
push `model/` + `radiant.js` in X-ray.web (radiant.js has the key and piece counts; the key stays the same while
`model_key.txt` exists). Each retrain adds ~300 MB to the website repo's history.

---

## The "X-ray or not" gate

`Radiant.analyse(file, show)` first runs `gate.onnx` (about 0.3 s). If the X-ray chance is below 0.5 it does **not**
run the big models and returns:
```javascript
{ gate: "not_xray", xray_chance: 0.0, message: "We believe this isn't an X-ray. Still run?" }
```
Show the message with a "Still run" button; on click call `Radiant.analyse(file, show, true)` (true = skip the gate).
Normal results also contain `gate` ("xray", or "skipped" after "Still run") and `xray_chance`.
Re-export the gate after retraining it: `python src/models/export_gate_onnx.py`.
