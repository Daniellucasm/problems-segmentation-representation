import cv2
import numpy as np
import os

base = os.path.dirname(os.path.dirname(__file__))
in_path = os.path.join(base, "data", "industrial.jpg")
out_dir = os.path.join(base, "results")

os.makedirs(out_dir, exist_ok=True)

img = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError(f"Não foi possível abrir {in_path}")

_, bin_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
contours = [c for c in contours if cv2.contourArea(c) > 200]

cnt = max(contours, key=cv2.contourArea)
peri = cv2.arcLength(cnt, True)

canvas = cv2.cvtColor(bin_img, cv2.COLOR_GRAY2BGR)

for frac in [0.005, 0.01, 0.02, 0.05]:
    eps = frac * peri
    approx = cv2.approxPolyDP(cnt, eps, True)
    cv2.polylines(canvas, [approx], True, (0, 255, 0), 2)
    print(f"Epsilon {frac*100:.1f}% → {len(approx)} vértices")

# Salvar resultado
out_path = os.path.join(out_dir, "industrial_poligonal.png")
cv2.imwrite(out_path, canvas)
print(f"Imagem salva em: {out_path}")
