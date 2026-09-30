import os, sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from ultralytics import YOLO

def main():
    m = YOLO('models/yolo11n/baseline/seed_0/weights/best.pt')
    res = m.val(data='configs/dmd_rgb.yaml', split='test', batch=32, workers=0, device=0, verbose=False)

    r_curve = None
    p_curve = None
    for cr in res.box.curves_results:
        x, y, xl, yl = cr
        if yl == 'Recall' and xl == 'Confidence':
            r_curve = (x, y)
        elif yl == 'Precision' and xl == 'Confidence':
            p_curve = (x, y)

    classes = ['yawning', 'hand_over_mouth', 'drinking', 'phone_use']
    for tau in [0.25, 0.50]:
        idx = np.argmin(np.abs(r_curve[0] - tau))
        actual_tau = r_curve[0][idx]
        rec_per_cls = {classes[i]: round(float(r_curve[1][i, idx]), 4) for i in range(4)}
        prec_per_cls = {classes[i]: round(float(p_curve[1][i, idx]), 4) for i in range(4)}
        macro_r = round(float(np.mean(list(rec_per_cls.values()))), 4)
        min_r = round(float(np.min(list(rec_per_cls.values()))), 4)
        macro_p = round(float(np.mean(list(prec_per_cls.values()))), 4)
        print(f'tau={tau:.2f} (sampled at {actual_tau:.3f}): Macro Recall={macro_r:.4f}, Min Recall={min_r:.4f}, Precision={macro_p:.4f}')
        print(f'  Per-Class Recall: {rec_per_cls}')

if __name__ == '__main__':
    main()
