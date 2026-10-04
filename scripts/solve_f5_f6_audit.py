"""F5 & F6 Audit Script.

F5:
- Scan README.md for curves and box.r claims.
- Recompute README Table 1 and Table 2 from canonical cache for all 30 pruned checkpoints + 8 complete baselines.
- Report all differences and flag any number changing by >0.5 pp.

F6:
- Decompose margin variance: pruning effect vs val->test shift vs tau* selection effect.
- Tabulate margin vs tau* rank correlation (Spearman rho and Pearson r).
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent
CLASS_NAMES = ['yawning', 'hand_over_mouth', 'drinking', 'phone_use']
DF_GT = pd.read_parquet(REPO_ROOT / 'results' / 'phase0' / 'ground_truth.parquet')

GT_VAL = DF_GT[DF_GT['split'] == 'val']
GT_TEST = DF_GT[DF_GT['split'] == 'test']
VAL_COUNTS = [len(GT_VAL[GT_VAL['class_id'] == c]) for c in range(4)]
TEST_COUNTS = [len(GT_TEST[GT_TEST['class_id'] == c]) for c in range(4)]

def get_class_recalls(df_dets, tau, gt_counts):
    recs, tps = [], []
    for c in range(4):
        tp = len(df_dets[(df_dets['class_id'] == c) & (df_dets['matched'] == True) & (df_dets['conf'] >= tau)])
        recs.append(tp / gt_counts[c] if gt_counts[c] > 0 else 1.0)
        tps.append(tp)
    return recs, tps

def compute_ap(recalls, precisions):
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([1.0], precisions, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    x = np.linspace(0, 1, 101)
    y = np.interp(x, mrec, mpre)
    return float(np.trapezoid(y, x) if hasattr(np, 'trapezoid') else np.trapz(y, x))

def get_eval_metrics(df_dets, split_gt):
    ap50_list = []
    for c_id in range(4):
        c_gt = split_gt[split_gt['class_id'] == c_id]
        n_gt = len(c_gt)
        if n_gt == 0:
            continue
        c_dets = df_dets[df_dets['class_id'] == c_id].sort_values(by='conf', ascending=False)
        if len(c_dets) == 0:
            ap50_list.append(0.0)
            continue
        tps = c_dets['matched'].values.astype(int)
        fps = 1 - tps
        tp_cum = np.cumsum(tps)
        fp_cum = np.cumsum(fps)
        prec = tp_cum / (tp_cum + fp_cum)
        rec = tp_cum / n_gt
        ap = compute_ap(rec, prec)
        ap50_list.append(ap)
    return ap50_list

def run_f5_f6():
    print("=" * 80)
    print("RUNNING F5 AND F6 AUDIT")
    print("=" * 80)

    # 1. Scan README for curves / box.r claims
    readme_path = REPO_ROOT / 'README.md'
    with open(readme_path, 'r', encoding='utf-8') as f:
        readme_text = f.read()

    print("\n--- Scanning README.md for curves / box.r claims ---")
    lines = readme_text.splitlines()
    flagged_lines = []
    for i, line in enumerate(lines):
        for term in ['r_curve', 'box.r', 'np.interp', 'interpolation', 'curves']:
            if term in line.lower():
                flagged_lines.append((i + 1, term, line))
    print(f"Found {len(flagged_lines)} lines mentioning curve/box.r terms in README.md:")
    for l_no, term, line in flagged_lines:
        print(f"  Line {l_no} [{term}]: {line.strip()[:100]}")

    # 2. Canonical evaluation for all 36 models (0-50% for yolo11n and yolo26n, seeds 0-2 for pruned, seeds 0-2/4 for baselines)
    models = ['yolo11n', 'yolo26n']
    ratios = ['0%', '10%', '20%', '30%', '40%', '50%']

    records = []

    for m in models:
        for r_str in ratios:
            seeds = [0, 1, 2] if r_str != '0%' else ([0, 1, 2, 3, 4] if m == 'yolo11n' else [0, 1, 2])
            for s in seeds:
                if r_str == '0%':
                    prefix = f"{m}_baseline_s{s}_best"
                else:
                    prefix = f"{m}_{r_str.replace('%', 'pct')}_s{s}_best"

                f_v = REPO_ROOT / 'results' / 'phase0' / 'cache' / f"{prefix}_val.parquet"
                f_t = REPO_ROOT / 'results' / 'phase0' / 'cache' / f"{prefix}_test.parquet"

                if not (f_v.exists() and f_t.exists()):
                    print(f"Missing cache for {prefix}")
                    continue

                df_v = pd.read_parquet(f_v)
                df_t = pd.read_parquet(f_t)

                # Validation metrics
                v_recs_25, _ = get_class_recalls(df_v, 0.25, VAL_COUNTS)
                v_min_25 = min(v_recs_25)
                v_floor = v_min_25 - 0.05

                # Exact tau* on validation
                unique_confs = np.sort(df_v['conf'].unique())[::-1]
                tau_star = None
                v_min_star = None
                for tau in unique_confs:
                    recs, _ = get_class_recalls(df_v, tau, VAL_COUNTS)
                    if min(recs) >= v_floor:
                        tau_star = float(tau)
                        v_min_star = min(recs)
                        break

                # Test evaluation at tau*
                t_recs_star, t_tps_star = get_class_recalls(df_t, tau_star, TEST_COUNTS)
                t_min_star = min(t_recs_star)
                margin_star = (t_min_star - v_floor) * 100.0
                compliance_star = bool(margin_star >= 0.0)

                # Macro precision at tau* on test
                sub_t = df_t[df_t['conf'] >= tau_star]
                precs = []
                for c in range(4):
                    c_sub = sub_t[sub_t['class_id'] == c]
                    tp = len(c_sub[c_sub['matched'] == True])
                    tot = len(c_sub)
                    precs.append(tp / tot if tot > 0 else 1.0)
                macro_prec_star = np.mean(precs)

                # AP metrics on test
                ap50s = get_eval_metrics(df_t, GT_TEST)
                min_ap50 = np.min(ap50s)
                m_ap50 = np.mean(ap50s)

                # Metrics at tau=0.25 on test
                t_recs_25, _ = get_class_recalls(df_t, 0.25, TEST_COUNTS)
                t_min_25 = min(t_recs_25)
                margin_25 = (t_min_25 - v_floor) * 100.0

                records.append({
                    'model': m, 'prune_ratio': r_str, 'seed': s,
                    'val_R_base': v_min_25, 'val_R_floor': v_floor,
                    'tau_star': tau_star, 'val_min_star': v_min_star,
                    'test_min_star': t_min_star, 'test_margin_star_pp': margin_star,
                    'passed_star': compliance_star, 'macro_prec_star': macro_prec_star,
                    'test_min_25': t_min_25, 'test_margin_25_pp': margin_25,
                    'yawn_ap50': ap50s[0], 'hom_ap50': ap50s[1],
                    'drink_ap50': ap50s[2], 'phone_ap50': ap50s[3],
                    'min_ap50': min_ap50, 'mAP50': m_ap50
                })

    df_all = pd.DataFrame(records)

    # 3. Aggregate Table 1 and Table 2 (using K=3 seeds 0-2 for apples-to-apples comparison with README)
    print("\n--- Comparing Canonical Recomputations against README Table 1 (K=3) ---")
    # Published values from README Table 1
    readme_table1_ref = {
        ('yolo11n', '0%'): {'tau_star': 0.672, 'val_min': 0.800, 'test_min': 0.552, 'margin': -23.7, 'mAP': 0.933},
        ('yolo11n', '10%'): {'tau_star': 0.645, 'val_min': 0.789, 'test_min': 0.653, 'margin': -13.6, 'mAP': 0.936},
        ('yolo11n', '20%'): {'tau_star': 0.637, 'val_min': 0.789, 'test_min': 0.673, 'margin': -11.6, 'mAP': 0.937},
        ('yolo11n', '30%'): {'tau_star': 0.639, 'val_min': 0.791, 'test_min': 0.690, 'margin': -9.9, 'mAP': 0.928},
        ('yolo11n', '40%'): {'tau_star': 0.512, 'val_min': 0.792, 'test_min': 0.711, 'margin': -7.8, 'mAP': 0.930},
        ('yolo11n', '50%'): {'tau_star': 0.505, 'val_min': 0.791, 'test_min': 0.632, 'margin': -15.7, 'mAP': 0.928},
        ('yolo26n', '0%'): {'tau_star': 0.540, 'val_min': 0.797, 'test_min': 0.709, 'margin': -8.7, 'mAP': 0.919},
        ('yolo26n', '10%'): {'tau_star': 0.625, 'val_min': 0.797, 'test_min': 0.608, 'margin': -18.9, 'mAP': 0.909},
        ('yolo26n', '20%'): {'tau_star': 0.546, 'val_min': 0.797, 'test_min': 0.601, 'margin': -19.6, 'mAP': 0.912},
        ('yolo26n', '30%'): {'tau_star': 0.557, 'val_min': 0.797, 'test_min': 0.640, 'margin': -15.6, 'mAP': 0.911},
        ('yolo26n', '40%'): {'tau_star': 0.614, 'val_min': 0.797, 'test_min': 0.657, 'margin': -13.9, 'mAP': 0.917},
        ('yolo26n', '50%'): {'tau_star': 0.459, 'val_min': 0.797, 'test_min': 0.728, 'margin': -6.9, 'mAP': 0.913},
    }

    diff_records = []
    # Evaluate across K=3 (seeds 0, 1, 2)
    df_k3 = df_all[df_all['seed'].isin([0, 1, 2])]

    for (m, r_str), grp in df_k3.groupby(['model', 'prune_ratio']):
        mean_tau_star = grp['tau_star'].mean()
        mean_val_min = grp['val_min_star'].mean()
        mean_test_min = grp['test_min_star'].mean()
        mean_margin = grp['test_margin_star_pp'].mean()
        mean_map = grp['mAP50'].mean()
        pass_rate = f"{grp['passed_star'].sum()}/{len(grp)}"

        ref = readme_table1_ref.get((m, r_str), {})
        ref_tau = ref.get('tau_star', np.nan)
        ref_margin = ref.get('margin', np.nan)
        ref_test_min = ref.get('test_min', np.nan)

        diff_tau = (mean_tau_star - ref_tau)
        diff_margin = (mean_margin - ref_margin)
        diff_test_min = (mean_test_min - ref_test_min) * 100.0

        is_flagged = abs(diff_margin) > 0.5 or abs(diff_test_min) > 0.5

        diff_records.append({
            'model': m, 'prune_ratio': r_str,
            'readme_tau_star': ref_tau, 'canon_tau_star': mean_tau_star, 'diff_tau': diff_tau,
            'readme_test_min': ref_test_min, 'canon_test_min': mean_test_min, 'diff_test_min_pp': diff_test_min,
            'readme_margin_pp': ref_margin, 'canon_margin_pp': mean_margin, 'diff_margin_pp': diff_margin,
            'canon_pass_rate': pass_rate, 'flag_gt_0.5pp': is_flagged
        })

    df_diffs = pd.DataFrame(diff_records)
    out_diffs = REPO_ROOT / 'results' / 'phase0' / 'f5_readme_discrepancies.csv'
    df_diffs.to_csv(out_diffs, index=False)
    print(f"\nDiscrepancies saved to {out_diffs}:")
    print(df_diffs[['model', 'prune_ratio', 'readme_margin_pp', 'canon_margin_pp', 'diff_margin_pp', 'flag_gt_0.5pp']].to_string(index=False))

    # 4. F6: Margin Variance Decomposition
    print("\n--- Executing F6 (Margin Variance Decomposition) ---")
    # Margin = Test_min(tau*) - R_floor
    # Decompose for each model:
    # Shift component: Test_min(0.25) - R_floor
    # Selection component: Test_min(tau*) - Test_min(0.25)
    df_all['shift_component_pp'] = df_all['test_margin_25_pp']
    df_all['selection_penalty_pp'] = (df_all['test_min_star'] - df_all['test_min_25']) * 100.0
    # Identity: test_margin_star_pp == shift_component_pp + selection_penalty_pp

    decomp_records = []
    for m in models:
        sub = df_all[df_all['model'] == m]
        var_total = float(np.var(sub['test_margin_star_pp'], ddof=1))
        var_shift = float(np.var(sub['shift_component_pp'], ddof=1))
        var_selection = float(np.var(sub['selection_penalty_pp'], ddof=1))
        cov_shift_sel = float(np.cov(sub['shift_component_pp'], sub['selection_penalty_pp'])[0, 1])

        # ANOVA over pruning ratios
        means_by_ratio = sub.groupby('prune_ratio')['test_margin_star_pp'].mean()
        grand_mean = sub['test_margin_star_pp'].mean()
        n_per_ratio = sub.groupby('prune_ratio')['test_margin_star_pp'].count()
        ss_between_ratios = float(np.sum(n_per_ratio * (means_by_ratio - grand_mean)**2))
        ss_total = float(np.sum((sub['test_margin_star_pp'] - grand_mean)**2))
        eta_sq_pruning = ss_between_ratios / ss_total if ss_total > 0 else 0.0

        # Correlation between tau* and margin
        tau_vals = sub['tau_star'].values
        margin_vals = sub['test_margin_star_pp'].values
        r_pearson, p_pearson = stats.pearsonr(tau_vals, margin_vals)
        r_spearman, p_spearman = stats.spearmanr(tau_vals, margin_vals)

        decomp_records.append({
            'model': m, 'n_runs': len(sub),
            'mean_margin_star_pp': sub['test_margin_star_pp'].mean(),
            'mean_shift_component_pp': sub['shift_component_pp'].mean(),
            'mean_selection_penalty_pp': sub['selection_penalty_pp'].mean(),
            'var_total_margin': var_total,
            'var_shift_component': var_shift,
            'var_selection_penalty': var_selection,
            'cov_shift_selection': cov_shift_sel,
            'pruning_eta_squared': eta_sq_pruning,
            'pruning_ss_share_pct': eta_sq_pruning * 100.0,
            'pearson_r_tau_margin': r_pearson, 'pearson_p': p_pearson,
            'spearman_rho_tau_margin': r_spearman, 'spearman_p': p_spearman
        })

    df_decomp = pd.DataFrame(decomp_records)
    out_decomp = REPO_ROOT / 'results' / 'phase0' / 'f6_margin_decomposition.csv'
    df_decomp.to_csv(out_decomp, index=False)
    print("\nMargin Variance Decomposition Table:")
    print(df_decomp.to_string(index=False))

    # Also save full df_all for downstream analyses
    out_all_ckpts = REPO_ROOT / 'results' / 'phase0' / 'all_evaluated_checkpoints_canonical.csv'
    df_all.to_csv(out_all_ckpts, index=False)
    print(f"\nAll evaluated checkpoints canonical data saved to {out_all_ckpts}")

if __name__ == '__main__':
    run_f5_f6()
