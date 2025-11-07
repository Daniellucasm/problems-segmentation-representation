import cv2
import numpy as np

def otsu_segment(gray, inv=False, blur_ks=0):
    img = gray.copy()
    if blur_ks > 0:
        img = cv2.GaussianBlur(img, (blur_ks, blur_ks), 0)
    flag = cv2.THRESH_BINARY_INV if inv else cv2.THRESH_BINARY
    thr, mask = cv2.threshold(img, 0, 255, flag + cv2.THRESH_OTSU)
    return mask, thr

if __name__ == "__main__":
    import argparse, os
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--inv", action="store_true")
    ap.add_argument("--blur", type=int, default=0)
    ap.add_argument("--out", default="results/otsu.png")
    args = ap.parse_args()

    gray = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    mask, thr = otsu_segment(gray, inv=args.inv, blur_ks=args.blur)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    cv2.imwrite(args.out, mask)
    print(f"Otsu threshold={thr:.2f} saved to {args.out}")
