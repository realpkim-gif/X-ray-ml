# How DINO finds the yellow box

DINO cuts the X-ray into a 32 x 32 grid of small squares and turns each square into a list of 768 numbers that
describe what's in it. Our small patch head was trained on X-rays where every square inside a vet's fracture
outline was marked "fracture" and every other square "healthy", so it gives each square a fracture probability:
that grid of probabilities is the red heatmap. The yellow box starts at the hottest square, grows into the touching
squares that are at least 30% as hot, and is drawn around that blob (each square = 32 pixels of the 1024-pixel image).
