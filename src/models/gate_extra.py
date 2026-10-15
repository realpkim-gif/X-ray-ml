import os
import io
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                      # draw charts without opening a window
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps, ImageEnhance
# ---------- always work from the project folder (X-ray-ml) ----------
# This file lives in src/<folder>/, but all paths below ("data/...", "src/models/weights/...") start at the
# project folder. These lines go there first, so it works from PyCharm's Run button AND from a terminal.
PROJECT_FOLDER = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # up 3: file -> src/<folder> -> src -> project
os.chdir(PROJECT_FOLDER)


# Extra pictures for the "X-ray or not" gate (Xray_Or_Not.py), made in code.
#
# Every picture TYPE is a small function make_<type>(rng, split) -> PIL image.
#   rng   = its own random number generator, so a different seed = different pictures
#   split = "train" / "val" / "test": X-ray types only use X-rays from that split (no leakage)
#
# NOT X-ray types (label 0): screenshots, documents, code, charts, tables, drawings, noise, memes, ...
# X-ray types (label 1):     X-rays changed in ways an upload might be: zoomed, rotated, inverted,
#                            photographed off a screen, tiny, with arrows/text, low contrast, ...
#
# add_training_data(type, counts) saves pictures to gate_extra/<label>/<type>/<split>/,
# and Xray_Or_Not.py automatically trains on everything in gate_extra/.
#
#   python src/models/gate_extra.py        (makes the first set: screenshot-like pictures)

EXTRA = "data/gate_extra"
FONTS = "C:/Windows/Fonts"
TEXT_FONTS = ["arial.ttf", "times.ttf", "calibri.ttf", "segoeui.ttf", "georgia.ttf", "verdana.ttf", "tahoma.ttf"]
CODE_FONTS = ["consola.ttf", "cour.ttf", "lucon.ttf"]
SCREEN_SIZES = [(1280, 720), (1920, 1080), (1366, 768), (800, 600), (1024, 768), (390, 844), (1080, 1920)]
WORDS = ("the of and to in is you that it he was for on are as with his they at be this have from or one had by "
         "word but not what all were we when your can said there use an each which she do how their if will up "
         "other about out many then them these so some her would make like him into time has look two more write go "
         "see number no way could people my than first water been call who oil its now find long down day did get "
         "come made may part patient report invoice total account settings profile login dashboard upload results").split()
CODE_WORDS = ["def", "return", "import", "for", "in", "if", "else", "while", "class", "self", "print", "range", "len",
              "model", "data", "x", "y", "loss", "torch", "np", "const", "let", "function", "await", "async", "=", "+",
              "(", ")", ":", "[", "]", "{", "}", "0", "1", "True", "None", "#", "//"]


def font(rng, names, size):
    for _ in range(5):                                   # some fonts may be missing on a machine
        try:
            return ImageFont.truetype(os.path.join(FONTS, rng.choice(names)), size)
        except OSError:
            pass
    return ImageFont.load_default()


def random_colour(rng, light):
    if light:
        return tuple(rng.randint(200, 255) for _ in range(3))
    return tuple(rng.randint(0, 70) for _ in range(3))


def sentence(rng, n):
    return " ".join(rng.choice(WORDS) for _ in range(n))


def xray_paths(split):
    # real X-rays of this split (no augmented copies)
    data = pd.read_csv(f"data/PreprocessedData/{split}.csv")
    return list(data[data["augmented"] == False]["image_path"])


# ======================================================================================
# NOT X-ray types (label 0)
# ======================================================================================

def make_text_page(rng, split):
    # a document / web article: title + paragraphs of text
    w, h = rng.choice(SCREEN_SIZES)
    light = rng.random() < 0.8
    bg = random_colour(rng, light)
    ink = random_colour(rng, not light)
    img = Image.new("RGB", (w, h), bg)
    d = ImageDraw.Draw(img)
    size = rng.randint(12, 30)
    y = rng.randint(10, 60)
    d.text((rng.randint(10, 80), y), sentence(rng, rng.randint(2, 6)).title(), fill=ink, font=font(rng, TEXT_FONTS, size * 2))
    y += size * 3
    f = font(rng, TEXT_FONTS, size)
    while y < h - size:
        d.text((rng.randint(10, 80), y), sentence(rng, max(3, w // (size * 4))), fill=ink, font=f)
        y += int(size * rng.uniform(1.3, 2.0))
    return img


def make_code(rng, split):
    # a code editor: line numbers + coloured code, dark or light theme
    w, h = rng.choice(SCREEN_SIZES)
    dark = rng.random() < 0.6
    img = Image.new("RGB", (w, h), (30, 30, 30) if dark else (250, 250, 250))
    d = ImageDraw.Draw(img)
    size = rng.randint(11, 22)
    f = font(rng, CODE_FONTS, size)
    colours = [(86, 156, 214), (206, 145, 120), (78, 201, 176), (220, 220, 170), (200, 200, 200) if dark else (40, 40, 40)]
    y, line = 8, 1
    while y < h - size:
        d.text((6, y), str(line), fill=(120, 120, 120), font=f)
        x = 50 + rng.randint(0, 4) * size * 2
        for _ in range(rng.randint(0, 8)):
            word = rng.choice(CODE_WORDS)
            d.text((x, y), word, fill=rng.choice(colours), font=f)
            x += (len(word) + 1) * size * 0.6
        y += int(size * 1.5)
        line += 1
    return img


def make_ui_layout(rng, split):
    # a website / app screen: top bar, side menu, cards, buttons, text
    w, h = rng.choice(SCREEN_SIZES)
    dark = rng.random() < 0.4
    img = Image.new("RGB", (w, h), random_colour(rng, not dark))
    d = ImageDraw.Draw(img)
    accent = tuple(rng.randint(30, 230) for _ in range(3))
    f = font(rng, TEXT_FONTS, rng.randint(12, 22))
    bar = rng.randint(40, 90)
    d.rectangle([0, 0, w, bar], fill=accent)
    d.text((15, bar // 3), sentence(rng, 2).title(), fill=(255, 255, 255), font=f)
    side = rng.randint(0, w // 4)
    if side > 30:
        d.rectangle([0, bar, side, h], fill=random_colour(rng, dark))
        for k in range(rng.randint(3, 10)):
            d.text((12, bar + 20 + k * 36), sentence(rng, 1).title(), fill=random_colour(rng, not dark), font=f)
    for _ in range(rng.randint(2, 9)):                    # cards with text and a button
        x1, y1 = rng.randint(side + 10, max(side + 11, w - 160)), rng.randint(bar + 10, max(bar + 11, h - 120))
        x2, y2 = min(w - 5, x1 + rng.randint(120, 420)), min(h - 5, y1 + rng.randint(80, 260))
        d.rounded_rectangle([x1, y1, x2, y2], radius=rng.randint(0, 16), fill=random_colour(rng, not dark),
                            outline=random_colour(rng, dark))
        d.text((x1 + 10, y1 + 10), sentence(rng, 3), fill=random_colour(rng, dark), font=f)
        if y2 - y1 > 70:
            d.rounded_rectangle([x1 + 10, y2 - 40, x1 + 110, y2 - 10], radius=6, fill=accent)
    return img


def make_chart(rng, split):
    # a chart: bars, lines, dots or pie, in a random matplotlib style
    plt.style.use(rng.choice(["default", "ggplot", "seaborn-v0_8", "dark_background", "bmh", "fivethirtyeight"]))
    fig, ax = plt.subplots(figsize=(rng.uniform(4, 10), rng.uniform(3, 7)))
    kind = rng.choice(["bar", "line", "scatter", "pie", "hist"])
    n = rng.randint(4, 30)
    values = []
    for k in range(n):
        values.append(rng.random() * 100)
    if kind == "bar":
        ax.bar(range(n), values)
    elif kind == "line":
        ax.plot(range(n), values, marker=rng.choice(["", "o", "s"]))
    elif kind == "scatter":
        other_values = []
        for k in range(n):
            other_values.append(rng.random() * 100)
        ax.scatter(values, other_values)
    elif kind == "pie":
        ax.pie(values[:6])
    else:
        bell_curve = []
        for k in range(300):
            bell_curve.append(rng.gauss(0, 1))
        ax.hist(bell_curve, bins=rng.randint(10, 40))
    ax.set_title(sentence(rng, 3).title())
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=rng.choice([80, 100, 130]))
    plt.close(fig)
    plt.style.use("default")
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def make_table(rng, split):
    # a spreadsheet / table: grid with numbers and words
    w, h = rng.choice(SCREEN_SIZES)
    img = Image.new("RGB", (w, h), (255, 255, 255))
    d = ImageDraw.Draw(img)
    f = font(rng, TEXT_FONTS, rng.randint(11, 18))
    cell_w, cell_h = rng.randint(70, 160), rng.randint(20, 34)
    header = tuple(rng.randint(150, 240) for _ in range(3))
    for row in range(h // cell_h + 1):
        for col in range(w // cell_w + 1):
            x, y = col * cell_w, row * cell_h
            d.rectangle([x, y, x + cell_w, y + cell_h], outline=(200, 200, 200), fill=header if row == 0 else None)
            text = sentence(rng, 1) if row == 0 or rng.random() < 0.3 else f"{rng.random() * 1000:.2f}"
            d.text((x + 4, y + 4), text, fill=(0, 0, 0), font=f)
    return img


def make_drawing(rng, split):
    # a simple drawing / clip-art: random shapes and lines on a plain background
    w, h = rng.choice([(512, 512), (800, 600), (600, 800)])
    img = Image.new("RGB", (w, h), random_colour(rng, rng.random() < 0.7))
    d = ImageDraw.Draw(img)
    for _ in range(rng.randint(3, 25)):
        box = sorted([rng.randint(0, w), rng.randint(0, w)]), sorted([rng.randint(0, h), rng.randint(0, h)])
        coords = [box[0][0], box[1][0], box[0][1], box[1][1]]
        colour = tuple(rng.randint(0, 255) for _ in range(3))
        shape = rng.choice(["ellipse", "rectangle", "line"])
        if shape == "ellipse":
            d.ellipse(coords, fill=colour if rng.random() < 0.5 else None, outline=colour, width=rng.randint(1, 6))
        elif shape == "rectangle":
            d.rectangle(coords, fill=colour if rng.random() < 0.5 else None, outline=colour, width=rng.randint(1, 6))
        else:
            d.line(coords, fill=colour, width=rng.randint(1, 10))
    return img


def make_noise(rng, split):
    # gradients / noise / blur: pictures with no real content
    w, h = rng.choice([(256, 256), (512, 512), (800, 600)])
    gradient = Image.linear_gradient("L").resize((w, h)).rotate(rng.randint(0, 359))
    noise = Image.effect_noise((w, h), rng.randint(10, 100))
    mixed = Image.blend(gradient, noise, rng.random())
    img = ImageOps.colorize(mixed, random_colour(rng, False), random_colour(rng, True))
    return img.filter(ImageFilter.GaussianBlur(rng.uniform(0, 6)))


# ======================================================================================
# X-ray types (label 1): real X-rays of the same split, changed like a real upload might be
# ======================================================================================

def open_xray(rng, split):
    return Image.open(rng.choice(xray_paths(split))).convert("RGB")


def make_xray_crop(rng, split):
    # a random zoomed piece (10-60% of the width)
    img = open_xray(rng, split)
    size = int(1024 * rng.uniform(0.1, 0.6))
    x, y = rng.randint(0, 1024 - size), rng.randint(0, 1024 - size)
    return img.crop((x, y, x + size, y + size))


def make_xray_rotated(rng, split):
    return open_xray(rng, split).rotate(rng.choice([90, 180, 270, rng.randint(-30, 30)]), expand=True)


def make_xray_inverted(rng, split):
    return ImageOps.invert(open_xray(rng, split))


def make_xray_small(rng, split):
    # a tiny thumbnail (100-300 px)
    img = open_xray(rng, split)
    size = rng.randint(100, 300)
    return img.resize((size, size))


def make_xray_low_contrast(rng, split):
    img = ImageEnhance.Contrast(open_xray(rng, split)).enhance(rng.uniform(0.3, 0.6))
    return ImageEnhance.Brightness(img).enhance(rng.uniform(0.7, 1.4))


def make_xray_annotated(rng, split):
    # a vet's notes on top: arrows, circles, text
    img = open_xray(rng, split)
    d = ImageDraw.Draw(img)
    colour = rng.choice([(255, 255, 0), (255, 0, 0), (0, 255, 0), (255, 255, 255)])
    for _ in range(rng.randint(1, 4)):
        x, y = rng.randint(100, 900), rng.randint(100, 900)
        d.ellipse([x - 60, y - 60, x + 60, y + 60], outline=colour, width=5)
        d.line([x + 60, y - 60, x + 160, y - 160], fill=colour, width=5)
    d.text((rng.randint(20, 600), rng.randint(20, 900)), sentence(rng, 3), fill=colour, font=font(rng, TEXT_FONTS, 40))
    return img


def make_xray_photo(rng, split):
    # a phone photo of an X-ray on a monitor / light box: tilted, tinted, blurred, darker edges
    img = open_xray(rng, split).resize((600, 600))
    tint = []                                                         # multiply red, green, blue a bit differently
    for k in range(3):
        tint.append(rng.uniform(0.7, 1.1))
    tint = np.array(tint)
    img = Image.fromarray(np.clip(np.asarray(img, dtype=np.float32) * tint, 0, 255).astype(np.uint8))
    background = Image.new("RGB", (800, 800), random_colour(rng, rng.random() < 0.3))
    background.paste(img, (rng.randint(20, 180), rng.randint(20, 180)))
    j = []                                                           # random tilt (perspective)
    for k in range(8):
        j.append(rng.randint(0, 80))
    quad = (j[0], j[1], j[2], 800 - j[3], 800 - j[4], 800 - j[5], 800 - j[6], j[7])
    img = background.transform((800, 800), Image.QUAD, quad)
    return img.filter(ImageFilter.GaussianBlur(rng.uniform(0, 2.5)))


def make_xray_tinted(rng, split):
    # an X-ray film scanned or photographed with a colour tint (blue, cyan, green or brown),
    # like many older / scanned X-rays on the internet.
    # TRIED overnight and NOT used: it did not fix the tinted web X-rays and made one photo a false alarm.
    gray = open_xray(rng, split).convert("L")
    gray = ImageEnhance.Contrast(gray).enhance(rng.uniform(0.5, 1.3))
    tints = [(0, 40, 90), (20, 90, 110), (10, 60, 40), (60, 40, 20)]       # colour of the darkest parts
    dark = rng.choice(tints)
    light = (rng.randint(200, 255), rng.randint(220, 255), rng.randint(220, 255))
    return ImageOps.colorize(gray, dark, light)                              # dark -> tint colour, light -> near white


TYPES = {
    # name: (function, label)
    "text_page": (make_text_page, 0), "code": (make_code, 0), "ui_layout": (make_ui_layout, 0),
    "chart": (make_chart, 0), "table": (make_table, 0), "drawing": (make_drawing, 0), "noise": (make_noise, 0),
    "xray_crop": (make_xray_crop, 1), "xray_rotated": (make_xray_rotated, 1), "xray_inverted": (make_xray_inverted, 1),
    "xray_small": (make_xray_small, 1), "xray_low_contrast": (make_xray_low_contrast, 1),
    "xray_annotated": (make_xray_annotated, 1), "xray_photo": (make_xray_photo, 1),
    "xray_tinted": (make_xray_tinted, 1),
}


def add_training_data(kind, counts, seed):
    # saves pictures of one type into gate_extra/<label>/<kind>/<split>/  (counts = {"train": 200, ...})
    make, label = TYPES[kind]
    rng = random.Random(seed)
    for split, count in counts.items():
        folder = f"{EXTRA}/{label}/{kind}/{split}"
        os.makedirs(folder, exist_ok=True)
        start = len(os.listdir(folder))                  # add to what's there, don't overwrite
        for i in range(count):
            make(rng, split).convert("RGB").save(f"{folder}/{kind}_{seed}_{start + i}.png")
    print("added", kind, counts)


if __name__ == "__main__":
    # the first set: screenshot-like pictures, 200 train / 40 val / 40 test of each = 1,000 train
    for k, kind in enumerate(["text_page", "code", "ui_layout", "chart", "table"]):
        add_training_data(kind, {"train": 200, "val": 40, "test": 40}, seed=1000 + k)
