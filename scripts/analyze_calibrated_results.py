"""Analyze and aggregate validation-calibrated evaluation sweep results."""

import json
import numpy as np

with open('results/calibrated_eval_data.json', 'r', encoding='utf-8') as f:
    d = json.load(f)

calib = d['calibration_info']
print("=" * 80)
print("1. VALIDATION BASELINE CALIBRATION (tau = 0.25, subjects 02, 03, 11)")
print("=" * 80)
for m in ['yolo11n', 'yolo26n']:
    c = calib[m]
    print(f"{m.upper()}: R_base^val = {c['r_base_val']:.4f} | R_floor = {c['r_floor']:.4f}")
    print(f"  Per-seed R_base: {[round(x, 4) for x in c['per_seed_r_base']]}")
    print(f"  Per-seed R_floor: {[round(x, 4) for x in c['per_seed_r_floor']]}")

print("\n" + "=" * 80)
print("2. OUT-OF-SAMPLE TEST EVALUATION AT VALIDATION-TUNED tau*")
print("=" * 80)

ratios = ['0%', '10%', '20%', '30%', '40%', '50%']
for m in ['yolo11n', 'yolo26n']:
    print(f"\n--- {m.upper()} ---")
    print(f"{'Pruning':<8} | {'tau*':<14} | {'Val MinR':<14} | {'Test MinR @ tau*':<18} | {'Achieved Margin':<16} | {'Compliance':<10} | {'Macro Prec':<14} | {'Min AP50':<14} | {'mAP50':<14}")
    print("-" * 135)
    for r in ratios:
        seeds = d[m][r]
        tau_stars = [seeds[s]['tau_star'] for s in ['0', '1', '2']]
        val_minrs = [seeds[s]['val_min_recall_at_tau_star'] for s in ['0', '1', '2']]
        test_minrs = [seeds[s]['test_at_tau_star']['min_recall'] for s in ['0', '1', '2']]
        margins = [seeds[s]['test_at_tau_star']['achieved_margin'] for s in ['0', '1', '2']]
        passes = [seeds[s]['test_at_tau_star']['passed_floor'] for s in ['0', '1', '2']]
        precs = [seeds[s]['test_at_tau_star']['macro_precision'] for s in ['0', '1', '2']]
        min_aps = [seeds[s]['test_min_ap50'] for s in ['0', '1', '2']]
        map50s = [seeds[s]['test_mAP50'] for s in ['0', '1', '2']]

        comp_rate = f"{sum(passes)}/3 ({sum(passes)/3*100:.0f}%)"
        t_str = f"{np.mean(tau_stars):.3f} ± {np.std(tau_stars, ddof=1):.3f}"
        vm_str = f"{np.mean(val_minrs):.3f} ± {np.std(val_minrs, ddof=1):.3f}"
        tm_str = f"{np.mean(test_minrs):.3f} ± {np.std(test_minrs, ddof=1):.3f}"
        mg_str = f"{np.mean(margins):+.3f} ± {np.std(margins, ddof=1):.3f}"
        p_str = f"{np.mean(precs):.3f} ± {np.std(precs, ddof=1):.3f}"
        ap_str = f"{np.mean(min_aps):.3f} ± {np.std(min_aps, ddof=1):.3f}"
        m50_str = f"{np.mean(map50s):.3f} ± {np.std(map50s, ddof=1):.3f}"

        print(f"{r:<8} | {t_str:<14} | {vm_str:<14} | {tm_str:<18} | {mg_str:<16} | {comp_rate:<10} | {p_str:<14} | {ap_str:<14} | {m50_str:<14}")
