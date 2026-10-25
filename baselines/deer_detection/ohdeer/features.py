"""Haar rectangle contrasts evaluated through a summed-area table.

No pretrained detector, CNN, HOG implementation, or cascade is used.
Each rectangle contributes its mean intensity; local-window standard
deviation normalizes contrast. Feature coordinates lie on a fixed grid.
"""
import numpy as np


def integral(image):
    return np.pad(np.asarray(image, dtype=np.float64).cumsum(0).cumsum(1),
                  ((1, 0), (1, 0)))


def rect_sum(ii, boxes):
    b = np.asarray(boxes, dtype=np.int64)
    x1, y1, x2, y2 = b.T
    return ii[y2, x2] - ii[y1, x2] - ii[y2, x1] + ii[y1, x1]


def make_features(count=256, grid=8, seed=566):
    """Fixed random two-, three-rectangle and checkerboard Haar templates."""
    if grid < 4 or count < 1:
        raise ValueError('grid >= 4 and count >= 1 required')
    rng = np.random.default_rng(seed)
    templates = []
    for _ in range(count):
        kind = int(rng.integers(5))
        nx, ny = [(2, 1), (1, 2), (3, 1), (1, 3), (2, 2)][kind]
        cw = int(rng.integers(1, grid // nx + 1))
        ch = int(rng.integers(1, grid // ny + 1))
        x = int(rng.integers(grid - nx * cw + 1))
        y = int(rng.integers(grid - ny * ch + 1))
        rects = []
        for j in range(ny):
            for i in range(nx):
                weight = (-2 if i == 1 else 1) if kind == 2 else (
                    (-2 if j == 1 else 1) if kind == 3 else (-1) ** (i+j))
                rects.append([x+i*cw, y+j*ch, x+(i+1)*cw, y+(j+1)*ch, weight])
        templates.append(rects)
    return {'grid': grid, 'templates': templates}


def extract(ii, ii_squared, windows, spec):
    """Return N x F features; all boxes are zero-based, half-open xyxy."""
    windows = np.asarray(windows, dtype=np.int64).reshape(-1, 4)
    if not len(windows):
        return np.empty((0, len(spec['templates'])), np.float32)
    x, y, right, bottom = windows.T
    width, height = right-x, bottom-y
    if (width < spec['grid']).any() or (height < spec['grid']).any():
        raise ValueError('Windows must be at least grid x grid')
    area = width * height
    mean = rect_sum(ii, windows) / area
    variance = np.maximum(rect_sum(ii_squared, windows) / area - mean**2, 0)
    contrast = np.maximum(np.sqrt(variance), 1 / 255)
    out = np.zeros((len(windows), len(spec['templates'])), np.float64)
    grid = spec['grid']
    for f, rects in enumerate(spec['templates']):
        for x1, y1, x2, y2, weight in rects:
            a = x + width*x1//grid
            b = y + height*y1//grid
            c = x + width*x2//grid
            d = y + height*y2//grid
            boxes = np.column_stack((a, b, c, d))
            out[:, f] += weight * rect_sum(ii, boxes) / ((c-a)*(d-b))
    return (out / contrast[:, None]).astype(np.float32)
