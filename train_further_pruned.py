"""Step 2: Prune and Fine-tune at 70%, 90%, 60%, 80% for YOLO11n and YOLO26n (Seeds 0, 1, 2).

Execution Order (70% and 90% for BOTH architectures first, then 60% and 80%):
1. YOLO11n 70% (Seeds 0, 1, 2)
2. YOLO26n 70% (Seeds 0, 1, 2)
3. YOLO11n 90% (Seeds 0, 1, 2)
4. YOLO26n 90% (Seeds 0, 1, 2)
5. YOLO11n 60% (Seeds 0, 1, 2)
6. YOLO26n 60% (Seeds 0, 1, 2)
7. YOLO11n 80% (Seeds 0, 1, 2)
8. YOLO26n 80% (Seeds 0, 1, 2)

Protocol: Fixed tau = 0.50, test split only, exact matching.
"""

import os
import sys
import time
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from ultralytics import YOLO

import prune.pruner
from eval.metrics import load_yolo_labels, match_detections_single_image, CLASS_NAMES
from plot_fixed_tau05 import generate_plots

REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / 'data' / 'processed' / 'RGB' / 'yolo'
CONFIG_YAML = REPO_ROOT / 'configs' / 'dmd_rgb.yaml'
RESULTS_DIR = REPO_ROOT / 'results'
FURTHER_CSV = RESULTS_DIR / 'fixed_tau05_checkpoints_further.csv'
BASE_0TO50_CSV = RESULTS_DIR / 'fixed_tau05_checkpoints_0to50.csv'
RESULTS_MD = REPO_ROOT / 'RESULTS_FIXED_TAU05.md'

# Execution sequence: 70% and 90% for BOTH architectures first, then 60% and 80%
RUN_SEQUENCE = [
    ('yolo11n', '70%'),
    ('yolo26n', '70%'),
    ('yolo11n', '90%'),
    ('yolo26n', '90%'),
    ('yolo11n', '60%'),
    ('yolo26n', '60%'),
    ('yolo11n', '80%'),
    ('yolo26n', '80%'),
]
SEEDS = [0, 1, 2]
TEST_GT = {0: 33, 1: 28, 2: 56, 3: 497}


def get_base_weights(model, seed):
    return REPO_ROOT / 'models' / model / 'baseline' / f'seed_{seed}' / 'weights' / 'best.pt'


def compute_bb_neck_params(model_obj, arch):
    m = model_obj.model
    total_p = sum(p.numel() for p in m.parameters())
    head_p = sum(p.numel() for p in m.model[-1].parameters())
    c2psa_p = sum(p.numel() for p in m.model[10].parameters())
    pre_head_p = 0
    if arch.upper() == 'YOLO26N':
        pre_head_p = sum(p.numel() for p in m.model[22].parameters())
    return total_p - (head_p + c2psa_p + pre_head_p)


def preload_test_annotations():
    txt_path = DATA_DIR / 'test.txt'
    gts = {}
    for line in open(txt_path):
        rel_p = line.strip()
        img_p = (DATA_DIR / rel_p).resolve()
        lbl_p = Path(str(img_p).replace('images', 'labels')).with_suffix('.txt')
        gts[str(img_p)] = load_yolo_labels(str(lbl_p), 640)
    return gts


def evaluate_checkpoint_tau05(weights_path, test_gts, arch):
    m = YOLO(str(weights_path))
    info = m.info(detailed=True)
    num_params = int(info[1])
    flops_g = float(info[3])
    # Size recorded strictly from final trained best.pt
    size_mb = Path(weights_path).stat().st_size / (1024 * 1024)
    bb_neck_p = compute_bb_neck_params(m, arch)

    txt_path = DATA_DIR / 'test.txt'
    res_gen = m.predict(source=str(txt_path), conf=0.001, batch=16, imgsz=640, device=0, verbose=False, stream=True)

    dets = []
    for r in res_gen:
        img_p = str(Path(r.path).resolve())
        boxes_gt = test_gts.get(img_p, [])
        gt_boxes = [(b[1], b[2], b[3], b[4]) for b in boxes_gt]
        gt_cls = [b[0] for b in boxes_gt]

        p_boxes = r.boxes.xyxy.cpu().numpy().tolist() if len(r.boxes) else []
        p_confs = r.boxes.conf.cpu().numpy().tolist() if len(r.boxes) else []
        p_cls = r.boxes.cls.cpu().int().numpy().tolist() if len(r.boxes) else []

        matches = match_detections_single_image(p_boxes, p_confs, p_cls, gt_boxes, gt_cls, iou_threshold=0.5)
        for conf, is_tp, cls_id in matches:
            if conf >= 0.50:
                dets.append({'conf': float(conf), 'is_tp': bool(is_tp), 'class_id': int(cls_id)})

    tot_det = len(dets)
    tp_tot = sum(1 for d in dets if d['is_tp'])
    prec = tp_tot / tot_det if tot_det > 0 else 0.0

    recs = {}
    for c, name in enumerate(CLASS_NAMES):
        tp_c = sum(1 for d in dets if d['is_tp'] and d['class_id'] == c)
        recs[name] = tp_c / TEST_GT[c]

    macro_r = float(np.mean(list(recs.values())))
    worst_r = float(min(recs.values()))

    return {
        'params': num_params,
        'bb_neck_params': bb_neck_p,
        'flops_g': flops_g,
        'size_mb': size_mb,
        'precision': prec,
        'yawning_rec': recs['yawning'],
        'hand_mouth_rec': recs['hand_over_mouth'],
        'drinking_rec': recs['drinking'],
        'phone_use_rec': recs['phone_use'],
        'macro_recall': macro_r,
        'worst_recall': worst_r,
        'status': 'COMPLETED'
    }


def update_results_markdown(all_records):
    df = pd.DataFrame(all_records)
    sparsity_order = ['0%', '10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%']

    lines = [
        "# Benchmark Results: Fixed Operating Point at $\\tau = 0.50$ (0% to 90% Sparsity)",
        "",
        "> **Protocol:** One fixed confidence threshold $\\tau = 0.50$ across every model, every seed, every sparsity level.  ",
        "> **Split:** Held-out test split (`subject_05`, `subject_10`, `subject_12`) only. Standard IoU=0.5 matching.  ",
        "> **Model Sizes:** Recorded strictly from trained `best.pt` checkpoints.  ",
        "",
        "---",
        "",
        "## 1. Summary Table: Mean ± SD across 3 Seeds (0% to 90% Sparsity)",
        "",
        "| Architecture | Sparsity | Total Params | BB+Neck Params | GFLOPs | Trained best.pt (MB) | Precision (%) | Yawning Rec (%) | Hand-Mouth Rec (%) | Drinking Rec (%) | Phone-Use Rec (%) | Macro-Recall (%) | Worst-Class Rec (%) | Achieved Param Red (%) |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for arch in ['YOLO11N', 'YOLO26N']:
        sub_arch = df[df['arch'] == arch]
        base_sub = sub_arch[sub_arch['sparsity'] == '0%']
        base_params = base_sub['params'].iloc[0] if len(base_sub) > 0 else 1

        for sp in sparsity_order:
            sub = sub_arch[sub_arch['sparsity'] == sp]
            if len(sub) == 0:
                continue

            p_mean = int(sub['params'].mean())
            bb_mean = int(sub['bb_neck_params'].mean()) if 'bb_neck_params' in sub.columns else 0
            f_mean = sub['flops_g'].mean()
            s_mean = sub['size_mb'].mean()

            prec_str = f"{sub['precision'].mean()*100:.1f} ± {sub['precision'].std()*100:.1f}%" if len(sub) > 1 else f"{sub['precision'].iloc[0]*100:.1f}%"
            yawn_str = f"{sub['yawning_rec'].mean()*100:.1f} ± {sub['yawning_rec'].std()*100:.1f}%" if len(sub) > 1 else f"{sub['yawning_rec'].iloc[0]*100:.1f}%"
            hand_str = f"{sub['hand_mouth_rec'].mean()*100:.1f} ± {sub['hand_mouth_rec'].std()*100:.1f}%" if len(sub) > 1 else f"{sub['hand_mouth_rec'].iloc[0]*100:.1f}%"
            drink_str = f"{sub['drinking_rec'].mean()*100:.1f} ± {sub['drinking_rec'].std()*100:.1f}%" if len(sub) > 1 else f"{sub['drinking_rec'].iloc[0]*100:.1f}%"
            phone_str = f"{sub['phone_use_rec'].mean()*100:.1f} ± {sub['phone_use_rec'].std()*100:.1f}%" if len(sub) > 1 else f"{sub['phone_use_rec'].iloc[0]*100:.1f}%"
            macro_str = f"{sub['macro_recall'].mean()*100:.1f} ± {sub['macro_recall'].std()*100:.1f}%" if len(sub) > 1 else f"{sub['macro_recall'].iloc[0]*100:.1f}%"
            worst_str = f"{sub['worst_recall'].mean()*100:.1f} ± {sub['worst_recall'].std()*100:.1f}%" if len(sub) > 1 else f"{sub['worst_recall'].iloc[0]*100:.1f}%"

            achieved_red = (1 - p_mean / base_params) * 100.0 if base_params > 0 else 0.0
            cond_str = "Baseline (0%)" if sp == '0%' else f"Pruned {sp}"

            lines.append(
                f"| {arch} | {cond_str} | {p_mean:,} | {bb_mean:,} | {f_mean:.2f} | {s_mean:.2f} | "
                f"{prec_str} | {yawn_str} | {hand_str} | {drink_str} | {phone_str} | {macro_str} | {worst_str} | {achieved_red:.1f}% |"
            )

    lines.extend([
        "",
        "---",
        "",
        "## 2. Per-Checkpoint Log",
        "",
        "| Arch | Sparsity | Seed | Total Params | BB+Neck Params | GFLOPs | Trained Size (MB) | Precision | Yawning | Hand-Mouth | Drinking | Phone-Use | Macro-Recall | Worst-Class Recall | Status |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ])

    for _, r in df.iterrows():
        cond_str = "Baseline" if r['sparsity'] == '0%' else f"Pruned {r['sparsity']}"
        bb_val = int(r['bb_neck_params']) if 'bb_neck_params' in r and not pd.isna(r['bb_neck_params']) else 0
        status_val = r.get('status', 'COMPLETED')
        lines.append(
            f"| {r['arch']} | {cond_str} | Seed {int(r['seed'])} | {int(r['params']):,} | {bb_val:,} | {r['flops_g']:.2f} | {r['size_mb']:.2f} | "
            f"{r['precision']*100:.1f}% | {r['yawning_rec']*100:.1f}% | {r['hand_mouth_rec']*100:.1f}% | "
            f"{r['drinking_rec']*100:.1f}% | {r['phone_use_rec']*100:.1f}% | {r['macro_recall']*100:.1f}% | {r['worst_recall']*100:.1f}% | {status_val} |"
        )

    with open(RESULTS_MD, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    try:
        generate_plots()
    except Exception as e:
        print(f"Plot update warning: {e}")


def verify_run_completion(target_dir, target_pt):
    """Check that results.csv has all 100 epoch rows and best.pt exists and is non-empty."""
    if not (target_pt.exists() and target_pt.stat().st_size > 500_000):
        return False, "best.pt missing or incomplete"
    res_csv = target_dir / 'results.csv'
    if not res_csv.exists():
        # Check parent or subdirectories
        alt_csv = list(target_dir.rglob('results.csv'))
        if alt_csv:
            res_csv = alt_csv[0]
        else:
            return False, "results.csv missing"
    try:
        df = pd.read_csv(res_csv)
        if len(df) < 100:
            return False, f"Incomplete epochs in results.csv: {len(df)}/100"
    except Exception as e:
        return False, f"Error reading results.csv: {e}"
    return True, "100 epochs verified"


def main():
    print("=" * 80)
    print("STARTING STEP 2: PRUNING & FINE-TUNING")
    print("Execution Order: 70% and 90% (both architectures), then 60% and 80%")
    print("Guardrails: 100-epoch completion check, trained best.pt size only, bb+neck params recorded.")
    print("=" * 80)

    test_gts = preload_test_annotations()
    base_records = pd.read_csv(BASE_0TO50_CSV).to_dict('records')
    further_records = []
    if FURTHER_CSV.exists() and FURTHER_CSV.stat().st_size > 0:
        further_records = pd.read_csv(FURTHER_CSV).to_dict('records')

    all_records = list(base_records) + list(further_records)
    update_results_markdown(all_records)

    total_runs = len(RUN_SEQUENCE) * len(SEEDS)
    current = 0

    for model, ratio in RUN_SEQUENCE:
        for seed in SEEDS:
            current += 1
            ratio_val = float(ratio.replace('%', '')) / 100.0
            label = ratio.replace('%', 'pct')
            key_str = f"{model.upper()} {ratio} (Seed {seed})"

            already = any(r['arch'] == model.upper() and r['sparsity'] == ratio and r['seed'] == seed for r in further_records)
            if already:
                print(f"[{current}/{total_runs}] {key_str} already completed. Skipping.")
                continue

            target_dir = REPO_ROOT / 'models' / model / 'pruning_fp32' / label / f'seed_{seed}'
            target_pt = target_dir / 'weights' / 'best.pt'
            base_pt = get_base_weights(model, seed)

            try:
                # Check if valid completed checkpoint already exists
                completed, reason = verify_run_completion(target_dir, target_pt)
                if not completed:
                    # Clean any partial/stale artifacts in run directory before launching training
                    if target_dir.exists():
                        # Strict deletion guard: ratio must be one of 60%, 70%, 80%, 90%
                        assert ratio in ['60%', '70%', '80%', '90%'], f"Refusing deletion: unauthorized ratio {ratio}"
                        assert label in ['60pct', '70pct', '80pct', '90pct'], f"Refusing deletion: unauthorized label {label}"
                        target_dir_str = str(target_dir.resolve()).replace('\\', '/')
                        assert any(f'/{pct}' in target_dir_str for pct in ['60pct', '70pct', '80pct', '90pct']), f"Target dir does not contain allowed ratio: {target_dir}"
                        assert 'baseline' not in target_dir_str, f"Forbidden: target dir contains 'baseline': {target_dir}"
                        assert not any(f'/{pct}' in target_dir_str for pct in ['0pct', '10pct', '20pct', '30pct', '40pct', '50pct']), f"Forbidden: target dir contains 0-50pct: {target_dir}"
                        shutil.rmtree(target_dir, ignore_errors=True)

                    pruned_dir = REPO_ROOT / 'models' / model / 'pruned' / label
                    pruned_pt = pruned_dir / f'seed_{seed}_pruned.pt'
                    pruned_dir.mkdir(parents=True, exist_ok=True)

                    print(f"\n[{current}/{total_runs}] Pruning {key_str} at ratio {ratio_val:.2f}...")
                    prune.pruner.prune_model(str(base_pt), ratio_val, str(pruned_pt))

                    print(f"Fine-tuning {key_str} for 100 epochs on CUDA:0...")
                    t0 = time.time()
                    ft_model = YOLO(str(pruned_pt))
                    train_results = ft_model.train(
                        trainer=prune.pruner.PrunedDetectionTrainer,
                        data=str(CONFIG_YAML),
                        epochs=100,
                        batch=16,
                        imgsz=640,
                        patience=0,
                        amp=True,
                        seed=seed,
                        project=str(target_dir.parent),
                        name=f'seed_{seed}',
                        exist_ok=True,
                        device=0,
                        workers=8,
                        cache=False,
                        verbose=False
                    )

                    candidates = [
                        target_pt,
                        target_dir / 'weights' / 'last.pt',
                    ]
                    if hasattr(ft_model, 'trainer') and ft_model.trainer and hasattr(ft_model.trainer, 'save_dir'):
                        candidates.append(Path(ft_model.trainer.save_dir) / 'weights' / 'best.pt')
                    if hasattr(train_results, 'save_dir'):
                        candidates.append(Path(train_results.save_dir) / 'weights' / 'best.pt')

                    saved = False
                    for c in candidates:
                        if c and Path(c).exists() and Path(c).stat().st_size > 500_000:
                            target_pt.parent.mkdir(parents=True, exist_ok=True)
                            if Path(c).resolve() != target_pt.resolve():
                                shutil.copy2(c, target_pt)
                            saved = True
                            break

                    # Strictly verify 100-epoch completion before evaluating
                    completed, reason = verify_run_completion(target_dir, target_pt)
                    if not completed:
                        print(f"WARNING: Run {key_str} was INCOMPLETE/CRASHED ({reason}).")
                        row = {
                            'arch': model.upper(), 'sparsity': ratio, 'seed': seed,
                            'params': 0, 'bb_neck_params': 0, 'flops_g': 0.0, 'size_mb': 0.0,
                            'precision': 0.0, 'yawning_rec': 0.0, 'hand_mouth_rec': 0.0,
                            'drinking_rec': 0.0, 'phone_use_rec': 0.0,
                            'macro_recall': 0.0, 'worst_recall': 0.0,
                            'status': 'INCOMPLETE/CRASHED'
                        }
                        further_records.append(row)
                        pd.DataFrame(further_records).to_csv(FURTHER_CSV, index=False)
                        all_records = list(base_records) + list(further_records)
                        update_results_markdown(all_records)
                        continue

                    print(f"Training completed in {(time.time() - t0)/60:.1f} minutes. 100 epochs verified. Saved to {target_pt}")

                # Evaluate verified checkpoint on test split at tau = 0.50
                print(f"Evaluating {key_str} on test split at tau=0.50...")
                ev = evaluate_checkpoint_tau05(target_pt, test_gts, arch=model.upper())
                ev['arch'] = model.upper()
                ev['sparsity'] = ratio
                ev['seed'] = seed

                further_records.append(ev)
                pd.DataFrame(further_records).to_csv(FURTHER_CSV, index=False)
                all_records = list(base_records) + list(further_records)
                update_results_markdown(all_records)

                print(
                    f"[{current}/{total_runs}] {key_str} | Prec={ev['precision']*100:.1f}% | "
                    f"MacroRec={ev['macro_recall']*100:.1f}% | Worst={ev['worst_recall']*100:.1f}% | "
                    f"Params={ev['params']:,} (BB+Neck={ev['bb_neck_params']:,}) | GFLOPs={ev['flops_g']:.2f} | Size={ev['size_mb']:.2f}MB"
                )
                sys.stdout.flush()

            except Exception as e:
                print(f"CRASH on {key_str}: {e}. Marking INCOMPLETE/CRASHED and continuing.")
                row = {
                    'arch': model.upper(), 'sparsity': ratio, 'seed': seed,
                    'params': 0, 'bb_neck_params': 0, 'flops_g': 0.0, 'size_mb': 0.0,
                    'precision': 0.0, 'yawning_rec': 0.0, 'hand_mouth_rec': 0.0,
                    'drinking_rec': 0.0, 'phone_use_rec': 0.0,
                    'macro_recall': 0.0, 'worst_recall': 0.0,
                    'status': 'INCOMPLETE/CRASHED'
                }
                further_records.append(row)
                pd.DataFrame(further_records).to_csv(FURTHER_CSV, index=False)
                all_records = list(base_records) + list(further_records)
                update_results_markdown(all_records)
                sys.stdout.flush()

    print("\nSTEP 2 COMPLETED!")


if __name__ == '__main__':
    main()
