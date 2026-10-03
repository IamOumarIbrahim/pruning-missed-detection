import json
import numpy as np
import pandas as pd
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CLASS_NAMES = ['yawning', 'hand_over_mouth', 'drinking', 'phone_use']
DF_GT = pd.read_parquet(REPO_ROOT / 'results' / 'phase0' / 'ground_truth.parquet')

def compute_ap(recalls, precisions):
    # Standard 101-point interpolation
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([1.0], precisions, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    x = np.linspace(0, 1, 101)
    return np.trapz(np.interp(x, mrec, mpre), x)

def eval_matched_metrics(df_dets, split_gt, n_images):
    # df_dets has cols: class_id, conf, matched, video, frame
    # Returns per-class and worst-class metrics
    results = {}
    
    # 1. Per-class AP50, Rec@P80, Rec@P90, Rec@FP05
    recs_p80 = []
    recs_p90 = []
    recs_fp05 = []
    ap50_list = []
    
    for c_id in range(4):
        c_gt = split_gt[split_gt['class_id'] == c_id]
        n_gt = len(c_gt)
        if n_gt == 0:
            continue
            
        c_dets = df_dets[df_dets['class_id'] == c_id].sort_values(by='conf', ascending=False)
        if len(c_dets) == 0:
            recs_p80.append(0.0)
            recs_p90.append(0.0)
            recs_fp05.append(0.0)
            ap50_list.append(0.0)
            continue
            
        tps = c_dets['matched'].values.astype(int)
        fps = (1 - tps)
        tp_cum = np.cumsum(tps)
        fp_cum = np.cumsum(fps)
        
        prec = tp_cum / (tp_cum + fp_cum)
        rec = tp_cum / n_gt
        fp_per_img = fp_cum / n_images
        
        ap = compute_ap(rec, prec)
        ap50_list.append(ap)
        
        # Rec@P80: max recall where precision >= 0.80
        valid_p80 = rec[prec >= 0.80]
        rec_p80 = float(np.max(valid_p80)) if len(valid_p80) > 0 else 0.0
        recs_p80.append(rec_p80)
        
        # Rec@P90: max recall where precision >= 0.90
        valid_p90 = rec[prec >= 0.90]
        rec_p90 = float(np.max(valid_p90)) if len(valid_p90) > 0 else 0.0
        recs_p90.append(rec_p90)
        
        # Rec@FP05: max recall where FP/img <= 0.05
        valid_fp = rec[fp_per_img <= 0.05]
        rec_fp = float(np.max(valid_fp)) if len(valid_fp) > 0 else 0.0
        recs_fp05.append(rec_fp)

    return {
        'mAP50': float(np.mean(ap50_list)),
        'min_class_AP50': float(np.min(ap50_list)),
        'worst_class_rec_p80': float(np.min(recs_p80)),
        'worst_class_rec_p90': float(np.min(recs_p90)),
        'worst_class_rec_fp05': float(np.min(recs_fp05)),
        'per_class_AP50': ap50_list,
        'per_class_rec_p80': recs_p80,
        'per_class_rec_p90': recs_p90,
        'per_class_rec_fp05': recs_fp05
    }

def run_r7_analysis():
    print("=" * 80)
    print("MANDATE R7: MATCHED OPERATING POINTS AND CLUSTER BOOTSTRAP")
    print("=" * 80)

    manifest_path = REPO_ROOT / 'results' / 'phase0' / 'cache_manifest.csv'
    if not manifest_path.exists():
        print("Manifest does not exist yet.")
        return

    df_manifest = pd.read_csv(manifest_path)
    ckpts = df_manifest[df_manifest['ckpt_type'] == 'best'].copy()

    # Image counts
    n_imgs = {
        'test': DF_GT[DF_GT['split'] == 'test']['image_id'].nunique(),
        'pooled': DF_GT[DF_GT['split'].isin(['val', 'test'])]['image_id'].nunique()
    }

    eval_records = []

    for (m, r_str, s), grp in ckpts.groupby(['model', 'prune_ratio', 'seed']):
        splits = grp['split'].tolist()
        if 'val' not in splits or 'test' not in splits:
            continue

        p_val = pd.read_parquet(REPO_ROOT / grp[grp['split'] == 'val']['parquet_path'].iloc[0])
        p_test = pd.read_parquet(REPO_ROOT / grp[grp['split'] == 'test']['parquet_path'].iloc[0])
        p_pool = pd.concat([p_val, p_test], ignore_index=True)

        gt_test = DF_GT[DF_GT['split'] == 'test']
        gt_pool = DF_GT[DF_GT['split'].isin(['val', 'test'])]

        m_test = eval_matched_metrics(p_test, gt_test, n_imgs['test'])
        m_pool = eval_matched_metrics(p_pool, gt_pool, n_imgs['pooled'])

        eval_records.append({
            'model': m, 'prune_ratio': r_str, 'seed': s,
            'test_mAP50': m_test['mAP50'],
            'test_min_AP50': m_test['min_class_AP50'],
            'test_worst_rec_p80': m_test['worst_class_rec_p80'],
            'test_worst_rec_p90': m_test['worst_class_rec_p90'],
            'test_worst_rec_fp05': m_test['worst_class_rec_fp05'],
            'pooled_mAP50': m_pool['mAP50'],
            'pooled_min_AP50': m_pool['min_class_AP50'],
            'pooled_worst_rec_p80': m_pool['worst_class_rec_p80'],
            'pooled_worst_rec_p90': m_pool['worst_class_rec_p90'],
            'pooled_worst_rec_fp05': m_pool['worst_class_rec_fp05'],
        })

    if not eval_records:
        print("No completed checkpoints ready for R7 yet.")
        return

    df_eval = pd.DataFrame(eval_records)
    out_csv = REPO_ROOT / 'results' / 'phase0' / 'r7_matched_operating_points.csv'
    df_eval.to_csv(out_csv, index=False)
    print(f"R7 metrics saved to: {out_csv}")
    print(df_eval.to_string())

if __name__ == '__main__':
    run_r7_analysis()
