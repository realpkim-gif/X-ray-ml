// =====================================================================================
// radiant.js - the whole fracture model, running INSIDE the browser (no server needed)
//
// It does the same as API.py, but in JavaScript with onnxruntime-web ("ort"):
//   1. stretch the X-ray to 1024 x 1024 (like the training X-rays)
//   2. cnn.onnx  : ConvNeXt  -> cnn vector [1, 1024]
//      dino.onnx : DINOv2    -> dino vector [1, 1536] + patches [1, 32, 32, 768]
//   3. heads.onnx: both heads, 5 Monte Carlo passes at once (random dropout masks) -> 10 votes + 5 heatmaps
//   4. majority vote = answer, share agreeing = confidence, average heatmap -> yellow box
//   5. draw the box and/or heatmap on the ORIGINAL picture
//
// How a web page uses it (see index.html):
//   <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.min.js"></script>
//   <script src="radiant.js"></script>
//   await Radiant.load("web_model/");                    // once: downloads the models (~200 MB, then cached)
//   const result = await Radiant.analyse(file, "both");  // file from <input type="file">
//   img.src = result.image;                              // the X-ray with box + heatmap drawn on it
//   img.src = Radiant.redraw(result, "box");             // switch view instantly, no re-running the model
// =====================================================================================

const Radiant = {};   // everything lives in this one object, so it can't clash with the website's own code

(function () {
    // ---------- settings ----------
    const HEATMAP_STRENGTH = 0.6;               // at most 60% red over a pixel (same as API.py)
    const BOX_COLOR = "rgb(255, 255, 0)";       // yellow
    const DISCLAIMER = "Not a diagnosis. Always have a veterinarian review the X-ray.";

    let config = null;                          // thresholds etc. from config.json
    let cnnSession = null, dinoSession = null, headsSession = null;   // the three loaded models

    // ---------- 1. load the models (call once) ----------
    Radiant.load = async function (modelFolder) {
        // modelFolder = where cnn.onnx, dino.onnx, heads.onnx, config.json are, e.g. "web_model/"
        // the WebAssembly engine files of onnxruntime come from the same CDN as ort.min.js
        ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
        config = await (await fetch(modelFolder + "config.json")).json();
        const options = { executionProviders: ["wasm"] };   // wasm = runs on the CPU, works in every browser
        cnnSession = await ort.InferenceSession.create(modelFolder + "cnn.onnx", options);
        dinoSession = await ort.InferenceSession.create(modelFolder + "dino.onnx", options);
        headsSession = await ort.InferenceSession.create(modelFolder + "heads.onnx", options);
    };

    // ---------- helpers ----------
    function canvasOf(width, height) {
        const canvas = document.createElement("canvas");    // an invisible drawing surface
        canvas.width = width;
        canvas.height = height;
        return canvas;
    }

    function toTensor(source, size) {
        // picture -> size x size -> numbers 0-1 in the order the model wants: [1, 3 colours, size, size]
        const canvas = canvasOf(size, size);
        const ctx = canvas.getContext("2d");
        ctx.imageSmoothingQuality = "high";                 // smooth shrinking (like PIL BILINEAR)
        ctx.drawImage(source, 0, 0, size, size);
        const rgba = ctx.getImageData(0, 0, size, size).data;   // [r, g, b, a, r, g, b, a, ...] 0-255
        const pixels = new Float32Array(3 * size * size);
        for (let i = 0; i < size * size; i++) {
            pixels[i] = rgba[i * 4] / 255;                         // all red values first
            pixels[size * size + i] = rgba[i * 4 + 1] / 255;       // then all green
            pixels[2 * size * size + i] = rgba[i * 4 + 2] / 255;   // then all blue
        }
        return new ort.Tensor("float32", pixels, [1, 3, size, size]);
    }

    function repeat(data, times) {
        // copy one row (e.g. the cnn vector) PASSES times -> one row per Monte Carlo pass
        const out = new Float32Array(data.length * times);
        for (let t = 0; t < times; t++) {
            out.set(data, t * data.length);          // copy the row into place number t
        }
        return out;
    }

    function randomMask(count, keep) {
        // Monte Carlo dropout: 1 = keep the neuron, 0 = drop it. Each one kept with chance "keep"
        const mask = new Float32Array(count);
        for (let i = 0; i < count; i++) {
            if (Math.random() < keep) {
                mask[i] = 1;
            } else {
                mask[i] = 0;
            }
        }
        return mask;
    }

    function round4(value) {
        // round to 4 decimals, e.g. 0.53125 -> 0.5313
        return Math.round(value * 10000) / 10000;
    }

    function findBox(heatmap) {
        // same as FractureLocator.find_box: start at the hottest square, grow into touching squares
        // at least box_threshold x as hot, box around that blob. Returns 0-1 fractions.
        const grid = config.grid;
        let best = 0;                                // number of the hottest square (row * grid + col)
        for (let i = 1; i < grid * grid; i++) {
            if (heatmap[i] > heatmap[best]) {
                best = i;
            }
        }
        const limit = config.box_threshold * heatmap[best];
        const inBlob = new Uint8Array(grid * grid);
        inBlob[best] = 1;
        const todo = [best];
        let minRow = grid, maxRow = 0, minCol = grid, maxCol = 0;
        while (todo.length > 0) {
            const i = todo.pop();
            const row = Math.floor(i / grid), col = i % grid;
            minRow = Math.min(minRow, row); maxRow = Math.max(maxRow, row);
            minCol = Math.min(minCol, col); maxCol = Math.max(maxCol, col);
            // the 4 touching squares: above, below, left, right
            const neighbourRows = [row - 1, row + 1, row, row];
            const neighbourCols = [col, col, col - 1, col + 1];
            for (let n = 0; n < 4; n++) {
                const r = neighbourRows[n];
                const c = neighbourCols[n];
                const insideGrid = r >= 0 && r < grid && c >= 0 && c < grid;
                if (insideGrid) {
                    const j = r * grid + c;
                    if (inBlob[j] === 0 && heatmap[j] >= limit) {
                        inBlob[j] = 1;
                        todo.push(j);
                    }
                }
            }
        }
        return {
            x1: round4(minCol / grid),
            y1: round4(minRow / grid),
            x2: round4((maxCol + 1) / grid),
            y2: round4((maxRow + 1) / grid),
        };
    }

    // ---------- 2. draw box and/or heatmap on the original picture ----------
    Radiant.redraw = function (result, show) {
        // show = "box", "heatmap" or "both". Returns a PNG data URL for <img src="...">.
        const original = result.original;
        const canvas = canvasOf(original.width, original.height);
        const ctx = canvas.getContext("2d");
        ctx.drawImage(original, 0, 0);

        if (show === "heatmap" || show === "both") {
            // 32 x 32 red squares, see-through by probability, then stretched smoothly over the X-ray
            const grid = config.grid;
            const small = canvasOf(grid, grid);
            const smallCtx = small.getContext("2d");
            const image = smallCtx.createImageData(grid, grid);
            for (let i = 0; i < grid * grid; i++) {
                const p = result.heatmap[Math.floor(i / grid)][i % grid];
                image.data[i * 4] = 255;                               // red
                image.data[i * 4 + 3] = Math.round(255 * p * HEATMAP_STRENGTH);   // how see-through
            }
            smallCtx.putImageData(image, 0, 0);
            ctx.imageSmoothingEnabled = true;                          // smooth stretching, like BILINEAR
            ctx.drawImage(small, 0, 0, original.width, original.height);
        }

        if ((show === "box" || show === "both") && result.box !== null) {   // no box for "normal"
            const b = result.box;
            ctx.strokeStyle = BOX_COLOR;
            ctx.lineWidth = Math.max(2, Math.round(Math.min(original.width, original.height) / 200));
            ctx.strokeRect(b.x1 * original.width, b.y1 * original.height,
                           (b.x2 - b.x1) * original.width, (b.y2 - b.y1) * original.height);
        }
        result.show = show;
        result.image = canvas.toDataURL("image/png");
        return result.image;
    };

    // ---------- 3. the whole pipeline for one X-ray ----------
    Radiant.analyse = async function (file, show = "both") {
        if (config === null) throw new Error("Call Radiant.load() first");
        const original = await createImageBitmap(file);        // decodes the uploaded picture

        // step 1: stretch to 1024 x 1024 like the training X-rays
        const base = canvasOf(1024, 1024);
        const baseCtx = base.getContext("2d");
        baseCtx.imageSmoothingQuality = "high";
        baseCtx.drawImage(original, 0, 0, 1024, 1024);

        // step 2: the two backbones
        const cnnOut = await cnnSession.run({ pixels: toTensor(base, 384) });
        const dinoOut = await dinoSession.run({ pixels: toTensor(base, 448) });

        // step 3: both heads, all PASSES at once (one row per pass, each with its own random masks)
        const n = config.passes;
        const headsOut = await headsSession.run({
            cnn: new ort.Tensor("float32", repeat(cnnOut.cnn.data, n), [n, 1024]),
            dino: new ort.Tensor("float32", repeat(dinoOut.dino.data, n), [n, 1536]),
            patches: new ort.Tensor("float32", repeat(dinoOut.patches.data, n), [n, 32, 32, 768]),
            keep_head: new ort.Tensor("float32", randomMask(n * 512, config.keep_head), [n, 512]),
            keep_patch: new ort.Tensor("float32", randomMask(n * 256, config.keep_patch), [n, 256, 1, 1]),
        });

        // step 4: 10 votes -> answer + confidence
        let votes = 0;
        for (let i = 0; i < n; i++) {
            if (headsOut.head_prob.data[i] >= config.head_threshold) votes++;
            if (headsOut.patch_prob.data[i] >= config.patch_threshold) votes++;
        }
        const share = votes / (2 * n);
        const isFracture = share >= 0.5;                       // 5-5 tie counts as fracture

        // average heatmap of the passes -> [32][32], and the box (only for fracture)
        const grid = config.grid;
        const mean = new Float32Array(grid * grid);          // the average heatmap, one number per square
        for (let i = 0; i < n * grid * grid; i++) {
            const square = i % (grid * grid);                 // which square this number belongs to
            mean[square] = mean[square] + headsOut.heatmap.data[i] / n;
        }
        const heatmap = [];                                   // the same, as 32 rows of 32 numbers
        for (let r = 0; r < grid; r++) {
            const row = [];
            for (let c = 0; c < grid; c++) {
                row.push(Math.round(mean[r * grid + c] * 1000) / 1000);
            }
            heatmap.push(row);
        }

        let answer = "normal";
        let box = null;                                       // null = no box (for "normal")
        if (isFracture) {
            answer = "fracture";
            box = findBox(mean);
        }
        const result = {
            answer: answer,
            confidence: Math.round(Math.max(share, 1 - share) * 100) / 100,
            fracture_votes: `${votes} of ${2 * n}`,
            box: box,
            heatmap: heatmap,
            show: show,
            image: null,
            disclaimer: DISCLAIMER,
            original: original,                                // kept so redraw() can switch views instantly
        };
        Radiant.redraw(result, show);                          // fills result.image
        return result;
    };
})();
