"""Master training pipeline for extending pruning sweep up to 90% and K=5 seeds.

Follows strict experimental redesign:
- Architectures: YOLO11n, YOLO26n
- Pruning ratios: 0% (baseline), 10%, 20%, 30%, 40%, 50%, 60%, 70%, 80%, 90%
- Seeds: 0, 1, 2, 3, 4 (K=5)
- Hyperparameters: batch=16, imgsz=640, epochs=100, amp=True, patience=0, workers=8
- Fault-tolerant: skips already completed runs, verifies checkpoint integrity, updates status JSON.
"""

import os
import sys
import json
import time
import shutil
import traceback
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from ultralytics import YOLO
import prune.pruner
from prune.pruner import prune_model, PrunedDetectionTrainer, get_model_info

DATA_YAML = str(REPO_ROOT / 'configs' / 'dmd_rgb.yaml')
STATUS_FILE = REPO_ROOT / 'results' / 'training_status.json'
LOG_FILE = REPO_ROOT / 'results' / 'training_deep_sweep.log'

EPOCHS = 100
BATCH = 16
IMGSZ = 640
WORKERS = 8
DEVICE = 0

MODELS = ['yolo11n', 'yolo26n']
ALL_SEEDS = [0, 1, 2, 3, 4]
ALL_RATIOS = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]


def log(msg):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    formatted = f"[{timestamp}] {msg}"
    print(formatted, flush=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(formatted + '\n')


def update_status(status_str, current_task=None, extra=None):
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'status': status_str,
        'last_updated': datetime.now().isoformat(),
        'current_task': current_task,
    }
    if extra:
        payload.update(extra)
    with open(STATUS_FILE, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)


def is_run_completed(model_name, ratio, seed):
    ratio_label = f'{int(ratio * 100)}pct'
    if ratio == 0.0:
        target_pt = REPO_ROOT / 'models' / model_name / 'baseline' / f'seed_{seed}' / 'weights' / 'best.pt'
    else:
        target_pt = REPO_ROOT / 'models' / model_name / 'pruning_fp32' / ratio_label / f'seed_{seed}' / 'weights' / 'best.pt'
    
    if target_pt.exists() and target_pt.stat().st_size > 1_000_000:
        return True
    return False


def get_baseline_weights(model_name, seed):
    target_pt = REPO_ROOT / 'models' / model_name / 'baseline' / f'seed_{seed}' / 'weights' / 'best.pt'
    if target_pt.exists() and target_pt.stat().st_size > 1_000_000:
        return str(target_pt)
    return None


def copy_best_weights(candidates, target_dest):
    target_dest = Path(target_dest)
    for c in candidates:
        if c and Path(c).exists() and Path(c).stat().st_size > 1_000_000:
            target_dest.parent.mkdir(parents=True, exist_ok=True)
            if Path(c).resolve() != target_dest.resolve():
                shutil.copy2(c, target_dest)
            return True
    return False


def train_baseline(model_name, seed):
    target_dir = REPO_ROOT / 'models' / model_name / 'baseline' / f'seed_{seed}'
    target_best = target_dir / 'weights' / 'best.pt'

    if is_run_completed(model_name, 0.0, seed):
        log(f"Baseline {model_name} seed {seed} already exists. Skipping.")
        return str(target_best)

    log(f"Training Baseline: {model_name} (seed {seed}) for {EPOCHS} epochs...")
    t0 = time.time()
    
    init_pt = REPO_ROOT / f'{model_name}.pt'
    model = YOLO(str(init_pt))
    
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        batch=BATCH,
        imgsz=IMGSZ,
        patience=0,
        amp=True,
        seed=seed,
        project=str(REPO_ROOT / 'models' / model_name / 'baseline'),
        name=f'seed_{seed}',
        exist_ok=True,
        device=DEVICE,
        workers=WORKERS,
        cache=False,
    )
    
    candidates = [
        target_best,
        target_dir / 'weights' / 'last.pt',
    ]
    if hasattr(model, 'trainer') and model.trainer and hasattr(model.trainer, 'save_dir'):
        s_dir = Path(model.trainer.save_dir) / 'weights'
        candidates.extend([s_dir / 'best.pt', s_dir / 'last.pt'])
    if hasattr(results, 'save_dir'):
        r_dir = Path(results.save_dir) / 'weights'
        candidates.extend([r_dir / 'best.pt', r_dir / 'last.pt'])
    runs_dir = REPO_ROOT / 'runs' / 'detect' / 'models' / model_name / 'baseline' / f'seed_{seed}' / 'weights'
    candidates.extend([runs_dir / 'best.pt', runs_dir / 'last.pt'])

    if not copy_best_weights(candidates, target_best):
        raise RuntimeError(f"Failed to find valid weights for {model_name} baseline seed {seed}")

    elapsed = (time.time() - t0) / 60
    log(f"Baseline {model_name} seed {seed} completed in {elapsed:.1f} min. Saved to {target_best}")
    return str(target_best)


def train_pruned(model_name, ratio, seed):
    ratio_label = f'{int(ratio * 100)}pct'
    target_dir = REPO_ROOT / 'models' / model_name / 'pruning_fp32' / ratio_label / f'seed_{seed}'
    target_best = target_dir / 'weights' / 'best.pt'

    if is_run_completed(model_name, ratio, seed):
        log(f"Pruned model {model_name} {ratio_label} seed {seed} already exists. Skipping.")
        return str(target_best)

    # Check baseline dependency
    baseline_w = get_baseline_weights(model_name, seed)
    if not baseline_w:
        log(f"Baseline missing for {model_name} seed {seed}. Training baseline first...")
        baseline_w = train_baseline(model_name, seed)

    # Prune model if pruned pt does not exist
    pruned_dir = REPO_ROOT / 'models' / model_name / 'pruned' / ratio_label
    pruned_path = pruned_dir / f'seed_{seed}_pruned.pt'
    if not pruned_path.exists() or pruned_path.stat().st_size < 100_000:
        log(f"Pruning {model_name} at {ratio:.0%} from {baseline_w}...")
        prune_model(baseline_w, ratio, str(pruned_path), imgsz=IMGSZ)
        info = get_model_info(str(pruned_path))
        log(f"Pruned checkpoint created: params={info['num_params']:,}, flops={info['flops_g']}G, size={info['size_mb']:.2f}MB")

    log(f"Fine-tuning {model_name} {ratio_label} (seed {seed}) for {EPOCHS} epochs...")
    t0 = time.time()
    
    ft_model = YOLO(str(pruned_path))
    results = ft_model.train(
        trainer=PrunedDetectionTrainer,
        data=DATA_YAML,
        epochs=EPOCHS,
        batch=BATCH,
        imgsz=IMGSZ,
        patience=0,
        amp=True,
        seed=seed,
        project=str(REPO_ROOT / 'models' / model_name / 'pruning_fp32' / ratio_label),
        name=f'seed_{seed}',
        exist_ok=True,
        device=DEVICE,
        workers=WORKERS,
        cache=False,
    )

    candidates = [
        target_best,
        target_dir / 'weights' / 'last.pt',
    ]
    if hasattr(ft_model, 'trainer') and ft_model.trainer and hasattr(ft_model.trainer, 'save_dir'):
        s_dir = Path(ft_model.trainer.save_dir) / 'weights'
        candidates.extend([s_dir / 'best.pt', s_dir / 'last.pt'])
    if hasattr(results, 'save_dir'):
        r_dir = Path(results.save_dir) / 'weights'
        candidates.extend([r_dir / 'best.pt', r_dir / 'last.pt'])
    runs_dir = REPO_ROOT / 'runs' / 'detect' / 'models' / model_name / 'pruning_fp32' / ratio_label / f'seed_{seed}' / 'weights'
    candidates.extend([runs_dir / 'best.pt', runs_dir / 'last.pt'])

    if not copy_best_weights(candidates, target_best):
        raise RuntimeError(f"Failed to find valid weights for {model_name} {ratio_label} seed {seed}")

    elapsed = (time.time() - t0) / 60
    log(f"Pruned fine-tuning {model_name} {ratio_label} seed {seed} completed in {elapsed:.1f} min. Saved to {target_best}")
    return str(target_best)


def build_task_queue():
    queue = []

    # Priority 1: Baseline seeds 3 & 4 (so all baselines exist for pruning)
    for m in MODELS:
        for s in [3, 4]:
            if not is_run_completed(m, 0.0, s):
                queue.append({'type': 'baseline', 'model': m, 'ratio': 0.0, 'seed': s})

    # Priority 2: Deep pruning ratios (60%, 70%, 80%, 90%) across seeds 0-4
    for r in [0.60, 0.70, 0.80, 0.90]:
        for m in MODELS:
            for s in ALL_SEEDS:
                if not is_run_completed(m, r, s):
                    queue.append({'type': 'pruned', 'model': m, 'ratio': r, 'seed': s})

    # Priority 3: Intermediate pruning ratios (10%, 20%, 30%, 40%, 50%) for seeds 3 & 4
    for r in [0.10, 0.20, 0.30, 0.40, 0.50]:
        for m in MODELS:
            for s in [3, 4]:
                if not is_run_completed(m, r, s):
                    queue.append({'type': 'pruned', 'model': m, 'ratio': r, 'seed': s})

    return queue


def main():
    log("=" * 60)
    log("Starting Master Training Pipeline: Deep Pruning (up to 90%) + K=5 Seeds")
    log("=" * 60)

    queue = build_task_queue()
    total_in_queue = len(queue)
    log(f"Identified {total_in_queue} pending training tasks.")

    if total_in_queue == 0:
        log("All 100 training tasks are already completed! Nothing to train.")
        update_status("COMPLETED", current_task=None, extra={'pending': 0, 'queue_length': 0})
        return

    update_status("RUNNING", current_task=None, extra={'total_pending': total_in_queue, 'queue': queue})

    completed_in_session = 0
    for idx, task in enumerate(queue):
        t_type = task['type']
        m = task['model']
        r = task['ratio']
        s = task['seed']
        r_pct = f'{int(r*100)}pct'
        task_desc = f"{m} {r_pct} seed {s}"

        log(f"\n>>> [Task {idx+1}/{total_in_queue}] Starting {task_desc} ({t_type})")
        update_status("RUNNING", current_task={
            'index': idx + 1,
            'total': total_in_queue,
            'task': task,
            'started_at': datetime.now().isoformat(),
        })

        try:
            if t_type == 'baseline':
                train_baseline(m, s)
            else:
                train_pruned(m, r, s)
            completed_in_session += 1
            log(f">>> [Task {idx+1}/{total_in_queue}] SUCCESS: {task_desc}")
        except Exception as e:
            log(f">>> [Task {idx+1}/{total_in_queue}] ERROR on {task_desc}: {e}")
            log(traceback.format_exc())
            update_status("ERROR", current_task={'task': task, 'error': str(e)})
            # Continue to next task or sleep briefly
            time.sleep(10)

    log("=" * 60)
    log(f"All training tasks finished. Completed {completed_in_session}/{total_in_queue} in this session.")
    log("=" * 60)
    update_status("COMPLETED", current_task=None, extra={'completed_in_session': completed_in_session})


if __name__ == '__main__':
    main()
