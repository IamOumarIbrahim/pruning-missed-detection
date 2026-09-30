"""Audit script to verify all 91 experiment runs and their artifacts."""

import json
from pathlib import Path

MODELS = ['yolo11n', 'yolo26n']
SEEDS = [0, 1, 2]
PRUNING_RATIOS = [10, 20, 30, 40, 50]
QUANT_LEVELS = ['fp16', 'int8', 'int4']
JOINT_RATIOS = [0, 10, 20, 30, 40, 50]

def audit():
    total_expected = 0
    total_found_metrics = 0
    total_found_models = 0
    missing_runs = []
    runs_details = []

    for model in MODELS:
        # 1. Baseline
        for seed in SEEDS:
            total_expected += 1
            m_path = Path(f'results/{model}/baseline/seed_{seed}/metrics.json')
            w_path = Path(f'models/{model}/baseline/seed_{seed}/weights/best.pt')
            m_ok = m_path.exists() and m_path.stat().st_size > 0
            w_ok = w_path.exists() and w_path.stat().st_size > 0
            if m_ok: total_found_metrics += 1
            if w_ok: total_found_models += 1
            if not (m_ok and w_ok):
                missing_runs.append((f'{model} baseline seed_{seed}', m_ok, w_ok))
            runs_details.append({
                'config': f'{model} baseline seed_{seed}',
                'metrics_ok': m_ok,
                'model_ok': w_ok,
                'model_path': str(w_path) if w_ok else None,
                'metrics_path': str(m_path) if m_ok else None,
            })

        # 2. Pruning FP32
        for r in PRUNING_RATIOS:
            for seed in SEEDS:
                total_expected += 1
                m_path = Path(f'results/{model}/pruning_fp32/{r}pct/seed_{seed}/metrics.json')
                w_path = Path(f'models/{model}/pruning_fp32/{r}pct/seed_{seed}/weights/best.pt')
                m_ok = m_path.exists() and m_path.stat().st_size > 0
                w_ok = w_path.exists() and w_path.stat().st_size > 0
                if m_ok: total_found_metrics += 1
                if w_ok: total_found_models += 1
                if not (m_ok and w_ok):
                    missing_runs.append((f'{model} pruning_{r}pct seed_{seed}', m_ok, w_ok))
                runs_details.append({
                    'config': f'{model} pruning_{r}pct seed_{seed}',
                    'metrics_ok': m_ok,
                    'model_ok': w_ok,
                    'model_path': str(w_path) if w_ok else None,
                    'metrics_path': str(m_path) if m_ok else None,
                })

        # 3. Quant only
        for q in QUANT_LEVELS:
            for seed in SEEDS:
                total_expected += 1
                m_path = Path(f'results/{model}/quant_only/{q}/seed_{seed}/metrics.json')
                w_path = Path(f'models/{model}/quant_only/{q}/seed_{seed}/model_{q}.onnx')
                m_ok = m_path.exists() and m_path.stat().st_size > 0
                w_ok = w_path.exists() and w_path.stat().st_size > 0
                if m_ok: total_found_metrics += 1
                if w_ok: total_found_models += 1
                if not (m_ok and w_ok):
                    missing_runs.append((f'{model} quant_{q} seed_{seed}', m_ok, w_ok))
                runs_details.append({
                    'config': f'{model} quant_{q} seed_{seed}',
                    'metrics_ok': m_ok,
                    'model_ok': w_ok,
                    'model_path': str(w_path) if w_ok else None,
                    'metrics_path': str(m_path) if m_ok else None,
                })

        # 4. Joint INT8
        for r in JOINT_RATIOS:
            for seed in SEEDS:
                total_expected += 1
                m_path = Path(f'results/{model}/joint_int8/{r}pct/seed_{seed}/metrics.json')
                w_path = Path(f'models/{model}/joint_int8/{r}pct/seed_{seed}/model_int8.onnx')
                m_ok = m_path.exists() and m_path.stat().st_size > 0
                w_ok = w_path.exists() and w_path.stat().st_size > 0
                if m_ok: total_found_metrics += 1
                if w_ok: total_found_models += 1
                if not (m_ok and w_ok):
                    missing_runs.append((f'{model} joint_{r}pct_int8 seed_{seed}', m_ok, w_ok))
                runs_details.append({
                    'config': f'{model} joint_{r}pct_int8 seed_{seed}',
                    'metrics_ok': m_ok,
                    'model_ok': w_ok,
                    'model_path': str(w_path) if w_ok else None,
                    'metrics_path': str(m_path) if m_ok else None,
                })

    # 5. QAT (YOLO11n 30% seed 0)
    total_expected += 1
    m_path = Path('results/yolo11n/qat/30pct/metrics.json')
    w_path = Path('models/yolo11n/qat/30pct/model_int8.onnx')
    m_ok = m_path.exists() and m_path.stat().st_size > 0
    w_ok = w_path.exists() and w_path.stat().st_size > 0
    if m_ok: total_found_metrics += 1
    if w_ok: total_found_models += 1
    if not (m_ok and w_ok):
        missing_runs.append(('yolo11n qat_30pct seed_0', m_ok, w_ok))
    runs_details.append({
        'config': 'yolo11n qat_30pct seed_0',
        'metrics_ok': m_ok,
        'model_ok': w_ok,
        'model_path': str(w_path) if w_ok else None,
        'metrics_path': str(m_path) if m_ok else None,
    })

    print(f'=== AUDIT SUMMARY ===')
    print(f'Total Expected Runs: {total_expected}')
    print(f'Metrics Found: {total_found_metrics}/{total_expected}')
    print(f'Model Files Found: {total_found_models}/{total_expected}')
    if missing_runs:
        print(f'Missing Runs ({len(missing_runs)}):')
        for name, m, w in missing_runs:
            print(f'  - {name}: metrics_exists={m}, model_exists={w}')
    else:
        print('ALL 91 RUNS AND CHECKPOINTS ARE FULLY COMPLETE AND VERIFIED!')

    # Specific check for yolo11n 5R/int8 seed 2
    spec_m = Path('results/yolo11n/joint_int8/50pct/seed_2/metrics.json')
    spec_w = Path('models/yolo11n/joint_int8/50pct/seed_2/model_int8.onnx')
    print(f'\nSpecific Check [yolo11n 5R/int8 seed 2]:')
    print(f'  metrics.json: {spec_m.exists()} ({spec_m.stat().st_size if spec_m.exists() else 0} bytes)')
    print(f'  model_int8.onnx: {spec_w.exists()} ({spec_w.stat().st_size if spec_w.exists() else 0} bytes)')

    return runs_details

if __name__ == '__main__':
    audit()
