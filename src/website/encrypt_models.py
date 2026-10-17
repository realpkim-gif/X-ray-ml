import os

PROJECT_FOLDER = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(PROJECT_FOLDER)   # go to the project folder, so the paths below work from anywhere

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

# encrypt_models.py - locks the browser model files so people can't just download and reuse them
#
# The website runs the model in the visitor's browser, so the browser has to download the weights.
# Plain .onnx files could be saved and reused by anyone. So we encrypt each file with AES-256-GCM
# (a strong, standard encryption) and radiant.js decrypts it in memory with the browser's
# built-in Web Crypto.
#
# Honest limit: the key has to be inside radiant.js (the browser needs it to decrypt), split in
# two scrambled halves. This stops people from just saving the files, but someone skilled who
# reads the JavaScript could still get the key. The README license is the legal protection.
#
# Each encrypted file = 12 random bytes (the "IV", a one-time number) + the encrypted bytes,
# cut into 45 MB pieces so GitHub Pages can host them. radiant.js glues the pieces back together.
# The file names don't say what's inside (part3-0.bin ... instead of dino.onnx).
#
# Run after export_onnx.py and export_gate_onnx.py:
#   python src/website/encrypt_models.py

ONNX_FOLDER = "src/models/weights/onnx"          # the plain files (made by the export scripts, never uploaded)
WEBSITE = "../X-ray.web"                          # the website repo, next to this project
OUT = WEBSITE + "/model"                          # encrypted files go here (committed to the website repo)
RADIANT_JS = WEBSITE + "/radiant.js"              # the key gets written into this file
KEY_FILE = "src/models/weights/model_key.txt"     # the key is kept here so it stays the same every run
PIECE_SIZE = 45 * 1000 * 1000                     # GitHub refuses files over 100 MB (warns over 50), so big files are cut into pieces

# plain file -> encrypted file name
FILES = [
    ["config.json", "part0.bin"],
    ["gate.onnx", "part1.bin"],
    ["cnn.onnx", "part2.bin"],
    ["dino.onnx", "part3.bin"],
    ["heads.onnx", "part4.bin"],
]

# 1. the key: 32 random bytes = 256 bits. Made once, then reused
if os.path.exists(KEY_FILE):
    key = bytes.fromhex(open(KEY_FILE).read().strip())
else:
    key = AESGCM.generate_key(bit_length=256)
    open(KEY_FILE, "w").write(key.hex())
    print("made a new key:", KEY_FILE)

aes = AESGCM(key)
os.makedirs(OUT, exist_ok=True)
for old in os.listdir(OUT):                      # remove the pieces of the last run
    os.remove(os.path.join(OUT, old))

# 2. encrypt every file, then cut it into pieces: part3.bin -> part3-0.bin, part3-1.bin, ...
pieces = []                                      # how many pieces each file has (radiant.js needs this)
for name, secret_name in FILES:
    data = open(os.path.join(ONNX_FOLDER, name), "rb").read()
    iv = os.urandom(12)                          # new random IV every time (never reuse one with the same key)
    locked = iv + aes.encrypt(iv, data, None)    # encrypted bytes + a 16-byte check that nobody changed them
    count = 0
    for start in range(0, len(locked), PIECE_SIZE):
        piece_name = secret_name.replace(".bin", "-" + str(count) + ".bin")
        open(os.path.join(OUT, piece_name), "wb").write(locked[start:start + PIECE_SIZE])
        count = count + 1
    pieces.append(count)
    print(name, "->", secret_name, round(len(data) / 1e6, 1), "MB,", count, "pieces")

# 3. write the key into radiant.js as two scrambled halves: key = KEY_A xor KEY_B
mask = os.urandom(32)
scrambled = bytearray()
for i in range(32):
    scrambled.append(key[i] ^ mask[i])

lines = open(RADIANT_JS, encoding="utf-8").read().split("\n")
found = 0
for i in range(len(lines)):
    if lines[i].strip().startswith("const KEY_A"):
        lines[i] = '    const KEY_A = "' + mask.hex() + '";'
        found = found + 1
    if lines[i].strip().startswith("const KEY_B"):
        lines[i] = '    const KEY_B = "' + scrambled.hex() + '";'
        found = found + 1
    if lines[i].strip().startswith("const PIECES"):
        lines[i] = '    const PIECES = ' + str(pieces) + ';   // pieces per file (written by encrypt_models.py)'
        found = found + 1
if found != 3:
    raise ValueError("could not find the KEY_A / KEY_B / PIECES lines in " + RADIANT_JS)
open(RADIANT_JS, "w", encoding="utf-8").write("\n".join(lines))
print("key written into", RADIANT_JS)
