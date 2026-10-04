import pandas as pd
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

df_t1_new = pd.read_csv(REPO_ROOT / 'results' / 'phase0' / 'readme_table1_regenerated.csv')
pub_t1 = [
    ('yolo11n', '0%', 0.672, 0.800, 0.552, -23.7, '0/3', 0.901, 0.915, 0.933),
    ('yolo11n', '10%', 0.645, 0.789, 0.653, -13.6, '0/3', 0.926, 0.910, 0.936),
    ('yolo11n', '20%', 0.637, 0.789, 0.673, -11.6, '0/3', 0.903, 0.909, 0.937),
    ('yolo11n', '30%', 0.639, 0.791, 0.690, -9.9, '0/3', 0.918, 0.897, 0.928),
    ('yolo11n', '40%', 0.512, 0.792, 0.711, -7.8, '1/3', 0.869, 0.901, 0.930),
    ('yolo11n', '50%', 0.505, 0.791, 0.632, -15.7, '1/3', 0.872, 0.898, 0.928),
    ('yolo26n', '0%', 0.540, 0.797, 0.709, -8.7, '1/3', 0.893, 0.880, 0.919),
    ('yolo26n', '10%', 0.625, 0.797, 0.608, -18.9, '0/3', 0.908, 0.865, 0.909),
    ('yolo26n', '20%', 0.546, 0.797, 0.601, -19.6, '0/3', 0.894, 0.854, 0.912),
    ('yolo26n', '30%', 0.557, 0.797, 0.640, -15.6, '0/3', 0.904, 0.856, 0.911),
    ('yolo26n', '40%', 0.614, 0.797, 0.657, -13.9, '0/3', 0.891, 0.882, 0.917),
    ('yolo26n', '50%', 0.459, 0.797, 0.728, -6.9, '0/3', 0.853, 0.855, 0.913),
]
df_pub = pd.DataFrame(pub_t1, columns=['Model', 'Pruning', 'pub_tau', 'pub_val_min', 'pub_test_min', 'pub_margin_pp', 'pub_comp', 'pub_prec', 'pub_min_ap', 'pub_map'])

print("=== TABLE 1 DISCREPANCIES (> 0.5 pp) ===")
changed_rows = 0
for i, r in df_t1_new.iterrows():
    p = df_pub[(df_pub['Model'] == r['Model']) & (df_pub['Pruning'] == r['Pruning'])].iloc[0]
    diffs = []
    if abs(r['tau_star_mean'] - p['pub_tau']) * 100 > 0.5:
        diffs.append(f"tau*: {p['pub_tau']:.3f} -> {r['tau_star_mean']:.3f} (diff {(r['tau_star_mean'] - p['pub_tau'])*100:+.2f} pp)")
    if abs(r['val_min_mean'] - p['pub_val_min']) * 100 > 0.5:
        diffs.append(f"val_min: {p['pub_val_min']:.3f} -> {r['val_min_mean']:.3f} (diff {(r['val_min_mean'] - p['pub_val_min'])*100:+.2f} pp)")
    if abs(r['test_min_mean'] - p['pub_test_min']) * 100 > 0.5:
        diffs.append(f"test_min: {p['pub_test_min']:.3f} -> {r['test_min_mean']:.3f} (diff {(r['test_min_mean'] - p['pub_test_min'])*100:+.2f} pp)")
    if abs(r['margin_mean'] - p['pub_margin_pp']) > 0.5:
        diffs.append(f"achieved_margin: {p['pub_margin_pp']:.1f} -> {r['margin_mean']:.1f} pp (diff {r['margin_mean'] - p['pub_margin_pp']:+.2f} pp)")
    if r['compliance'].split()[0] != p['pub_comp']:
        diffs.append(f"compliance: {p['pub_comp']} -> {r['compliance']}")
    if abs(r['macro_prec_mean'] - p['pub_prec']) * 100 > 0.5:
        diffs.append(f"macro_prec: {p['pub_prec']:.3f} -> {r['macro_prec_mean']:.3f} (diff {(r['macro_prec_mean'] - p['pub_prec'])*100:+.2f} pp)")
    if abs(r['min_ap50_mean'] - p['pub_min_ap']) * 100 > 0.5:
        diffs.append(f"min_ap50: {p['pub_min_ap']:.3f} -> {r['min_ap50_mean']:.3f} (diff {(r['min_ap50_mean'] - p['pub_min_ap'])*100:+.2f} pp)")
    if abs(r['mAP50_mean'] - p['pub_map']) * 100 > 0.5:
        diffs.append(f"mAP50: {p['pub_map']:.3f} -> {r['mAP50_mean']:.3f} (diff {(r['mAP50_mean'] - p['pub_map'])*100:+.2f} pp)")
    if diffs:
        changed_rows += 1
        print(f"\n{r['Model']} {r['Pruning']}:")
        for d in diffs:
            print(f"   - {d}")

print(f"\nTotal rows with discrepancies > 0.5 pp: {changed_rows} / {len(df_t1_new)}")
