"""Detection metrics: IoU matching, per-class recall, precision."""

import numpy as np
from pathlib import Path
from ultralytics import YOLO

CLASS_NAMES = ['yawning', 'hand_over_mouth', 'drinking', 'phone_use']
NUM_CLASSES = len(CLASS_NAMES)


def load_yolo_labels(label_path, img_size=640):
    """Load YOLO-format labels and convert to xyxy pixel coordinates.

    Returns list of (class_id, x1, y1, x2, y2) tuples.
    """
    boxes = []
    path = Path(label_path)
    if not path.exists() or path.stat().st_size == 0:
        return boxes
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            cx, cy, w, h = (float(parts[1]), float(parts[2]),
                            float(parts[3]), float(parts[4]))
            x1 = (cx - w / 2) * img_size
            y1 = (cy - h / 2) * img_size
            x2 = (cx + w / 2) * img_size
            y2 = (cy + h / 2) * img_size
            boxes.append((cls, x1, y1, x2, y2))
    return boxes


def compute_iou_matrix(boxes_a, boxes_b):
    """IoU matrix between two sets of xyxy boxes.

    Args:
        boxes_a: (N, 4) array.
        boxes_b: (M, 4) array.

    Returns:
        (N, M) IoU matrix.
    """
    x1 = np.maximum(boxes_a[:, 0:1], boxes_b[:, 0:1].T)
    y1 = np.maximum(boxes_a[:, 1:2], boxes_b[:, 1:2].T)
    x2 = np.minimum(boxes_a[:, 2:3], boxes_b[:, 2:3].T)
    y2 = np.minimum(boxes_a[:, 3:4], boxes_b[:, 3:4].T)
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / (union + 1e-7)


def match_detections_single_image(pred_boxes, pred_confs, pred_classes,
                                  gt_boxes, gt_classes, iou_threshold=0.5):
    """Greedy IoU matching for one image.

    Returns list of (confidence, is_tp, pred_class_id).
    """
    matches = []
    if len(gt_boxes) == 0:
        for conf, cls in zip(pred_confs, pred_classes):
            matches.append((float(conf), False, int(cls)))
        return matches
    if len(pred_boxes) == 0:
        return matches

    pred_boxes_np = np.array(pred_boxes)
    gt_boxes_np = np.array(gt_boxes)
    iou_matrix = compute_iou_matrix(pred_boxes_np, gt_boxes_np)

    order = np.argsort(-np.array(pred_confs))
    matched_gt = set()

    for idx in order:
        conf = pred_confs[idx]
        pred_cls = pred_classes[idx]
        best_iou = 0.0
        best_gt = -1
        for gt_idx in range(len(gt_boxes)):
            if gt_idx in matched_gt:
                continue
            if gt_classes[gt_idx] != pred_cls:
                continue
            if iou_matrix[idx, gt_idx] > best_iou:
                best_iou = iou_matrix[idx, gt_idx]
                best_gt = gt_idx
        if best_iou >= iou_threshold and best_gt >= 0:
            matches.append((float(conf), True, int(pred_cls)))
            matched_gt.add(best_gt)
        else:
            matches.append((float(conf), False, int(pred_cls)))
    return matches


def compute_detection_metrics(matches, gt_counts, threshold,
                              class_names=None):
    """Per-class recall, precision, and safety metrics at a threshold.

    Args:
        matches: list of (confidence, is_tp, class_id).
        gt_counts: dict class_id -> count.
        threshold: confidence threshold.
        class_names: list of class names.

    Returns:
        dict with per_class_recall, macro_recall, min_safety_recall,
        precision, total_tp, total_fp.
    """
    if class_names is None:
        class_names = CLASS_NAMES
    filtered = [(c, tp, cls) for c, tp, cls in matches if c >= threshold]

    tp_per_class = {i: 0 for i in range(len(class_names))}
    total_per_class = {i: 0 for i in range(len(class_names))}
    for _, is_tp, cls in filtered:
        if cls in tp_per_class:
            total_per_class[cls] += 1
            if is_tp:
                tp_per_class[cls] += 1

    recall_per_class = {}
    for cls_id, name in enumerate(class_names):
        gt = gt_counts.get(cls_id, 0)
        recall_per_class[name] = tp_per_class[cls_id] / gt if gt > 0 else 1.0

    total_tp = sum(tp_per_class.values())
    total_pred = sum(total_per_class.values())
    precision = total_tp / total_pred if total_pred > 0 else 0.0

    recalls = list(recall_per_class.values())
    macro_recall = float(np.mean(recalls)) if recalls else 0.0
    min_safety_recall = float(min(recalls)) if recalls else 0.0

    return {
        'per_class_recall': recall_per_class,
        'macro_recall': macro_recall,
        'min_safety_recall': min_safety_recall,
        'precision': float(precision),
        'total_tp': total_tp,
        'total_fp': total_pred - total_tp,
    }


def evaluate_model(model, data_yaml, split='test', imgsz=640):
    """Full evaluation: mAP via Ultralytics val, plus raw matches.

    Args:
        model: YOLO object or path to .pt / .engine.
        data_yaml: Path to dataset YAML.
        split: 'val' or 'test'.
        imgsz: Input size.

    Returns:
        dict with map50, map50_95, matches, gt_counts, image_count.
    """
    import yaml as _yaml

    if isinstance(model, (str, Path)):
        model = YOLO(str(model))

    # mAP via Ultralytics val
    val_results = model.val(data=str(data_yaml), split=split, imgsz=imgsz,
                            batch=1, verbose=False)
    map50 = float(val_results.box.map50)
    map50_95 = float(val_results.box.map)

    # Resolve image list
    with open(data_yaml) as f:
        data_cfg = _yaml.safe_load(f)
    data_root = Path(data_yaml).parent / data_cfg['path']
    split_file = data_root / data_cfg.get(split, data_cfg.get('test'))

    with open(split_file) as f:
        image_paths = [l.strip() for l in f if l.strip()]
    txt_dir = split_file.parent
    resolved = [(txt_dir / p).resolve() for p in image_paths]

    # Run predictions at low conf to capture all detections
    all_matches = []
    gt_class_counts = {i: 0 for i in range(NUM_CLASSES)}

    results_gen = model.predict(source=resolved, conf=0.001, iou=0.5,
                                imgsz=imgsz, verbose=False, stream=True)
    for img_path, result in zip(resolved, results_gen):
        if len(result.boxes) > 0:
            pred_boxes = result.boxes.xyxy.cpu().numpy().tolist()
            pred_confs = result.boxes.conf.cpu().numpy().tolist()
            pred_classes = result.boxes.cls.cpu().int().numpy().tolist()
        else:
            pred_boxes, pred_confs, pred_classes = [], [], []

        label_path = Path(
            str(img_path).replace('/images/', '/labels/')
                         .replace('\\images\\', '\\labels\\')
        ).with_suffix('.txt')
        gt_raw = load_yolo_labels(label_path, img_size=imgsz)
        gt_boxes = [(b[1], b[2], b[3], b[4]) for b in gt_raw]
        gt_classes = [b[0] for b in gt_raw]
        for cls in gt_classes:
            gt_class_counts[cls] = gt_class_counts.get(cls, 0) + 1

        img_matches = match_detections_single_image(
            pred_boxes, pred_confs, pred_classes,
            gt_boxes, gt_classes, iou_threshold=0.5,
        )
        all_matches.extend(img_matches)

    return {
        'map50': map50,
        'map50_95': map50_95,
        'matches': all_matches,
        'gt_counts': gt_class_counts,
        'image_count': len(resolved),
    }
