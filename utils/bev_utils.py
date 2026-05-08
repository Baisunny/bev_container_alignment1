import numpy as np

def phys_to_bev(phys_coords, x_min, x_max, y_min, y_max, res):
    coords = np.asarray(phys_coords, dtype=np.float32)
    u = (coords[:, 0] - x_min) / res
    v = (coords[:, 1] - y_min) / res
    return np.stack([u, v], axis=1)

def generate_gaussian_heatmap(centers, grid_shape, sigma=2.0):
    H, W = grid_shape
    num = len(centers)
    heatmap = np.zeros((num, H, W), dtype=np.float32)
    s = float(sigma)
    for i, (u, v) in enumerate(centers):
        if not np.isfinite(u) or not np.isfinite(v):
            continue
        x0 = int(np.floor(u - 3 * s))
        x1 = int(np.ceil(u + 3 * s))
        y0 = int(np.floor(v - 3 * s))
        y1 = int(np.ceil(v + 3 * s))
        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(W - 1, x1)
        y1 = min(H - 1, y1)
        if x0 > x1 or y0 > y1:
            continue
        xs = np.arange(x0, x1 + 1, dtype=np.float32)
        ys = np.arange(y0, y1 + 1, dtype=np.float32)
        yy, xx = np.meshgrid(ys, xs, indexing="ij")
        g = np.exp(-((xx - u) ** 2 + (yy - v) ** 2) / (2 * s ** 2))
        patch = heatmap[i, y0 : y1 + 1, x0 : x1 + 1]
        np.maximum(patch, g, out=patch)
    return heatmap
