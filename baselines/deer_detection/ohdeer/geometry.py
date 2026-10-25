import numpy as np


def iou(box, boxes):
    box = np.asarray(box, dtype=float)
    boxes = np.asarray(boxes, float).reshape(-1, 4)
    if not len(boxes):
        return np.empty(0)
    top_left = np.maximum(boxes[:, :2], np.asarray(box)[:2])
    bottom_right = np.minimum(boxes[:, 2:], np.asarray(box)[2:])
    wh = np.maximum(bottom_right-top_left, 0)
    intersection = wh[:, 0] * wh[:, 1]
    a = np.maximum(boxes[:, 2]-boxes[:, 0], 0) * np.maximum(boxes[:, 3]-boxes[:, 1], 0)
    b = max(box[2]-box[0], 0) * max(box[3]-box[1], 0)
    return intersection / np.maximum(a+b-intersection, 1e-12)


def negative_is_clean(box, ground_truth, max_iou=0.05, max_coverage=0.05):
    """Reject partially covered deer too: low IoU alone is insufficient."""
    gt = np.asarray(ground_truth, float).reshape(-1, 4)
    if not len(gt):
        return True
    wh = np.maximum(np.minimum(gt[:, 2:], box[2:]) - np.maximum(gt[:, :2], box[:2]), 0)
    intersection = wh[:, 0] * wh[:, 1]
    coverage = intersection / max((box[2]-box[0])*(box[3]-box[1]), 1)
    return bool((iou(box, gt) <= max_iou).all() and (coverage <= max_coverage).all())


def nms(boxes, scores, threshold=0.3, limit=100):
    boxes = np.asarray(boxes, float).reshape(-1, 4)
    scores = np.asarray(scores)
    order = np.argsort(-scores, kind='stable')
    keep = []
    while len(order) and len(keep) < limit:
        j = int(order[0]); keep.append(j)
        remaining = order[1:]
        order = remaining[iou(boxes[j], boxes[remaining]) <= threshold]
    return np.asarray(keep, dtype=np.int64)


def windows(width, height, fractions, ratios, stride_fraction, min_size=16):
    """Multi-size windows on the image, including right/bottom boundaries."""
    sizes = set()
    for fraction in fractions:
        w = int(round(width * fraction))
        for ratio in ratios:
            h = int(round(w / ratio))
            if min_size <= w <= width and min_size <= h <= height:
                sizes.add((w, h))
    for w, h in sorted(sizes):
        stride = max(1, int(round(min(w, h) * stride_fraction)))
        xs = sorted(set(range(0, width-w+1, stride)) | {width-w})
        ys = sorted(set(range(0, height-h+1, stride)) | {height-h})
        for y in ys:
            for x in xs:
                yield (x, y, x+w, y+h)
