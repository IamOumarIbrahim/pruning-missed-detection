"""Validation + test evaluation of the co-design crop models at fixed tau = 0.50.

Evaluates the deployed PTQ INT8 ONNX model of every (crop size, seed) on
  - the validation split of the ROI partition
  - the held-out test split (3,213 images, same list the benchmark used)
with the same protocol as run_codesign_multiseed.py: confidence threshold
tau = 0.50, greedy bipartite matching at IoU = 0.50.

Writes results/codesign_val_test.md and results/codesign_val_test.csv.
Put this file in scripts/ and run:  python scripts/eval_codesign_val_test.py
"""

import argparse
import hashlib
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from eval.metrics import load_yolo_labels, match_detections_single_image, CLASS_NAMES

TAU = 0.50
SIZES = [320, 160]
SEEDS = [72, 73, 74]
SPLITS = ['val', 'test']
DEFAULT_ROI_ROOT = Path(r'C:\Dev\repos\Public repos\research\ROI-DMS\partitions\cropped')
EXPECTED_TEST_GT = {0: 33, 1: 28, 2: 56, 3: 451}
IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp'}
CSV_TOL = 2e-3


def collect_split(size, split, roi_root):
    """Return (images, gt dict keyed by resolved path, per-class GT counts)."""
    split_dir = roi_root / f'{size}x{size}' / split
    list_file = REPO_ROOT / 'data' / f'roi{size}_{split}.txt'
    if list_file.exists():
        images = [Path(line.strip()) for line in open(list_file, encoding='utf-8') if line.strip()]
    else:
        img_dir = split_dir / 'images'
        if not img_dir.exists():
            raise FileNotFoundError(f'Image folder not found: {img_dir}')
        images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT)
    if not images:
        raise FileNotFoundError(f'No images found for {size}x{size} {split}')

    gts = {}
    counts = {c: 0 for c in range(len(CLASS_NAMES))}
    for img in images:
        lbl = split_dir / 'labels' / f'{img.stem}.txt'
        boxes = load_yolo_labels(str(lbl), size) if lbl.exists() else []
        gts[str(img.resolve())] = boxes
        for b in boxes:
            counts[int(b[0])] += 1
    return images, gts, counts


def evaluate(model_path, images, gts, counts, size):
    """Same protocol as run_codesign_multiseed.evaluate_model, plus per-class recall."""
    m = YOLO(str(model_path), task='detect')
    res_gen = m.predict(source=[str(p) for p in images], conf=0.001, imgsz=size,
                        device='cpu', verbose=False, stream=True)
    dets = []
    n_seen = 0
    t0 = time.time()
    for r in res_gen:
        n_seen += 1
        boxes_gt = gts.get(str(Path(r.path).resolve()), [])
        gt_boxes = [(b[1], b[2], b[3], b[4]) for b in boxes_gt]
        gt_cls = [b[0] for b in boxes_gt]
        has = len(r.boxes) > 0
        p_boxes = r.boxes.xyxy.cpu().numpy().tolist() if has else []
        p_confs = r.boxes.conf.cpu().numpy().tolist() if has else []
        p_cls = r.boxes.cls.cpu().int().numpy().tolist() if has else []
        for conf, is_tp, cls_id in match_detections_single_image(
                p_boxes, p_confs, p_cls, gt_boxes, gt_cls, iou_threshold=0.5):
            if conf >= TAU:
                dets.append((bool(is_tp), int(cls_id)))
    elapsed = time.time() - t0

    tp_total = sum(1 for tp, _ in dets if tp)
    precision = tp_total / len(dets) if dets else 0.0
    recalls = {}
    for c, name in enumerate(CLASS_NAMES):
        tp_c = sum(1 for tp, cid in dets if tp and cid == c)
        recalls[name] = tp_c / counts[c] if counts[c] > 0 else 0.0
    return {
        'precision': precision,
        'macro': float(np.mean(list(recalls.values()))),
        'worst': float(min(recalls.values())),
        'recalls': recalls,
        'n_images': n_seen,
        'n_expected': len(images),
        'seconds': elapsed,
    }


def md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def pct(x):
    return f'{x * 100:.1f}%'


def mean_sd(vals):
    a = np.array(vals, dtype=float)
    if len(a) == 1:
        return pct(a[0])
    return f'{a.mean() * 100:.1f} ± {a.std(ddof=1) * 100:.1f}%'


def csv_check(size, seed, test_res, csv_df):
    if csv_df is None:
        return 'no CSV'
    sub = csv_df[(csv_df['model'] == 'YOLO11N') & (csv_df['input_res'] == f'{size}x{size} Crop')
                 & (csv_df['seed'] == seed)]
    if sub.empty:
        return 'no CSV row'
    r = sub.iloc[-1]
    pairs = [('precision', 'precision'), ('macro_recall', 'macro'), ('worst_recall', 'worst')]
    if all(abs(float(r[k]) - test_res[v]) <= CSV_TOL for k, v in pairs):
        return 'match'
    return f"MISMATCH (csv {r['precision']:.4f}/{r['macro_recall']:.4f}/{r['worst_recall']:.4f})"


def main():
    ap = argparse.ArgumentParser(description='Val + test evaluation of co-design crop models')
    ap.add_argument('--roi-root', default=str(DEFAULT_ROI_ROOT))
    ap.add_argument('--sizes', nargs='+', type=int, default=SIZES)
    ap.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    ap.add_argument('--out', default=str(REPO_ROOT / 'results' / 'codesign_val_test.md'))
    args = ap.parse_args()
    roi_root = Path(args.roi_root)
    out_md = Path(args.out)
    out_csv = out_md.with_suffix('.csv')

    csv_path = REPO_ROOT / 'results' / 'codesign_benchmark.csv'
    csv_df = pd.read_csv(csv_path) if csv_path.exists() and csv_path.stat().st_size > 0 else None

    # Load every split once per size
    data = {}
    notes = []
    for size in args.sizes:
        for split in SPLITS:
            images, gts, counts = collect_split(size, split, roi_root)
            data[(size, split)] = (images, gts, counts)
            print(f'{size}x{size} {split}: {len(images)} images, GT counts {counts}')
        test_counts = data[(size, 'test')][2]
        if test_counts != EXPECTED_TEST_GT:
            notes.append(f'WARNING: {size}x{size} test GT counts {test_counts} differ from '
                         f'the counts hard-coded in run_codesign_multiseed.py {EXPECTED_TEST_GT}.')

    rows = []
    file_hashes = {}
    integrity = []
    for size in args.sizes:
        for seed in args.seeds:
            base = REPO_ROOT / 'models' / 'codesign' / f'yolo11n_{size}_pruned70' / f'seed_{seed}'
            onnx = base / 'ptq' / 'model_int8.onnx'
            if not onnx.exists():
                msg = f'{size}x{size} seed {seed}: {onnx} not found, skipped.'
                print(msg)
                notes.append(f'WARNING: {msg}')
                continue
            h = md5(onnx)
            file_hashes[(size, seed)] = h
            res_csv = base / 'train' / 'results.csv'
            epochs = len(pd.read_csv(res_csv)) if res_csv.exists() else None
            for split in SPLITS:
                images, gts, counts = data[(size, split)]
                print(f'Evaluating {size}x{size} seed {seed} on {split} ...')
                r = evaluate(onnx, images, gts, counts, size)
                if r['n_images'] != r['n_expected']:
                    notes.append(f'WARNING: {size}x{size} seed {seed} {split}: predicted on '
                                 f'{r["n_images"]} of {r["n_expected"]} images.')
                print(f'  prec {pct(r["precision"])}  macro {pct(r["macro"])}  worst {pct(r["worst"])}  '
                      f'({r["seconds"]:.0f}s)')
                row = {'size': size, 'seed': seed, 'split': split, 'epochs': epochs,
                       'precision': r['precision'], 'macro_recall': r['macro'],
                       'worst_recall': r['worst'], 'onnx_md5': h}
                for name in CLASS_NAMES:
                    row[f'recall_{name}'] = r['recalls'][name]
                if split == 'test':
                    integrity.append((size, seed, h, csv_check(size, seed, r, csv_df)))
                rows.append(row)

    if not rows:
        raise SystemExit('Nothing was evaluated. Check that the ONNX files exist.')

    df = pd.DataFrame(rows)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    # Duplicate-model check
    for size in args.sizes:
        hs = [h for (s, _), h in file_hashes.items() if s == size]
        if len(hs) > 1 and len(set(hs)) < len(hs):
            notes.append(f'WARNING: {size}x{size} has identical ONNX files across seeds '
                         f'({len(set(hs))} unique of {len(hs)}). The seeds are not independent.')

    L = [
        '# Co-Design Crop Models: Validation and Test Evaluation',
        '',
        f'> Generated {datetime.now():%Y-%m-%d %H:%M}. Model: YOLO11N, 70% pruned, PTQ INT8 ONNX (`ptq/model_int8.onnx`).  ',
        f'> Protocol: fixed confidence threshold tau = {TAU:.2f}, greedy bipartite matching at IoU = 0.50, '
        'CPU ONNX Runtime inference.  ',
        '> Validation split: the `val` split of the ROI partition. Test split: the held-out list '
        '`data/roi<size>_test.txt` used by the benchmark.',
        '',
    ]
    if notes:
        L += ['## Warnings', ''] + [f'- {n}' for n in notes] + ['']

    L += ['## 1. Split Sizes and Ground-Truth Counts', '',
          '| Crop | Split | Images | ' + ' | '.join(CLASS_NAMES) + ' |',
          '| :--- | :--- | :--- | ' + ' | '.join(':---' for _ in CLASS_NAMES) + ' |']
    for size in args.sizes:
        for split in SPLITS:
            images, _, counts = data[(size, split)]
            L.append(f'| {size}x{size} | {split} | {len(images):,} | ' +
                     ' | '.join(str(counts[c]) for c in range(len(CLASS_NAMES))) + ' |')

    L += ['', '## 2. Summary: Mean ± SD across Seeds', '',
          '| Crop | Split | Seeds | Precision | Macro-Recall | Worst-Class Recall |',
          '| :--- | :--- | :--- | :--- | :--- | :--- |']
    for size in args.sizes:
        for split in SPLITS:
            sub = df[(df['size'] == size) & (df['split'] == split)]
            if sub.empty:
                continue
            L.append(f'| {size}x{size} | {split} | {len(sub)} | {mean_sd(sub["precision"])} | '
                     f'{mean_sd(sub["macro_recall"])} | {mean_sd(sub["worst_recall"])} |')

    L += ['', '## 3. Per-Seed Results', '',
          '| Crop | Seed | Split | Epochs | Precision | Macro-Recall | Worst-Class Recall | ' +
          ' | '.join(f'{n} R' for n in CLASS_NAMES) + ' |',
          '| :--- | :--- | :--- | :--- | :--- | :--- | :--- | ' + ' | '.join(':---' for _ in CLASS_NAMES) + ' |']
    for _, r in df.sort_values(['size', 'seed', 'split'], ascending=[False, True, False]).iterrows():
        ep = '' if pd.isna(r['epochs']) else int(r['epochs'])
        L.append(f'| {int(r["size"])}x{int(r["size"])} | {int(r["seed"])} | {r["split"]} | {ep} | '
                 f'{pct(r["precision"])} | {pct(r["macro_recall"])} | {pct(r["worst_recall"])} | ' +
                 ' | '.join(pct(r[f'recall_{n}']) for n in CLASS_NAMES) + ' |')

    L += ['', '## 4. Integrity Checks', '',
          '| Crop | Seed | ONNX md5 | Test result vs `codesign_benchmark.csv` |',
          '| :--- | :--- | :--- | :--- |']
    for size, seed, h, chk in sorted(integrity, key=lambda t: (-t[0], t[1])):
        L.append(f'| {size}x{size} | {seed} | `{h}` | {chk} |')
    L += ['', '---', '*Generated by `scripts/eval_codesign_val_test.py`.*']

    out_md.write_text('\n'.join(L), encoding='utf-8')
    print(f'\nWrote {out_md}\nWrote {out_csv}')
    for n in notes:
        print(n)


if __name__ == '__main__':
    main()
