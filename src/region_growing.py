import cv2
import numpy as np
from collections import deque

def region_growing(gray, seeds, tau=15, connectivity=8):
    h, w = gray.shape
    visited = np.zeros_like(gray, dtype=bool)
    mask = np.zeros_like(gray, dtype=np.uint8)
    if connectivity == 8:
        neigh = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    else:
        neigh = [(-1,0),(0,-1),(0,1),(1,0)]
    for (sy, sx) in seeds:
        if visited[sy, sx]: 
            continue
        q = deque([(sy, sx)])
        visited[sy, sx] = True
        region_pixels = [(sy, sx)]
        region_mean = float(gray[sy, sx])
        while q:
            y, x = q.popleft()
            for dy, dx in neigh:
                ny, nx = y+dy, x+dx
                if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx]:
                    if abs(int(gray[ny, nx]) - region_mean) <= tau:
                        visited[ny, nx] = True
                        q.append((ny, nx))
                        region_pixels.append((ny, nx))
                        # atualizar média incremental
                        region_mean += (int(gray[ny, nx]) - region_mean) / len(region_pixels)
        for (y, x) in region_pixels:
            mask[y, x] = 255
    return mask

if __name__ == "__main__":
    import argparse, json, os
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--seeds", required=True, help="JSON: [[y,x], [y,x], ...]")
    ap.add_argument("--tau", type=int, default=15)
    ap.add_argument("--conn", type=int, default=8)
    ap.add_argument("--out", default="results/rg.png")
    args = ap.parse_args()

    gray = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    seeds = json.loads(args.seeds)
    mask = region_growing(gray, seeds, tau=args.tau, connectivity=args.conn)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    cv2.imwrite(args.out, mask)
    print(f"RG tau={args.tau} saved to {args.out}")
