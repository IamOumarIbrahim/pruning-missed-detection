import json
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent
CLASS_NAMES = ['yawning', 'hand_over_mouth', 'drinking', 'phone_use']
GT_COUNTS = {
    'val': [34, 30, 68, 517],
    'test': [33, 28, 56, 497]
}

def analyze_model_tau(model_path, split, tau):
    m = YOLO(model_path)
    
    # Run val at conf=tau
    res_tau = m.val(
        data='configs/dmd_rgb.yaml',
        split=split,
        batch=64,
        device=0,
        workers=0,
        conf=tau,
        iou=0.7,
        max_det=300,
        plots=False,
        verbose=False,
        project='runs/detect/r2_diag',
        name=f'run_tau_{int(tau*1000)}',
        exist_ok=True
    )
    
    # What does Ultralytics report?
    # 1. res_tau.box.r (this is at max F1)
    ultra_box_r = list(res_tau.box.r) if hasattr(res_tau.box, 'r') else []
    
    # What is the max F1 confidence in res_tau?
    px = res_tau.box.px
    f1_curve = res_tau.box.f1_curve
    from ultralytics.utils.metrics import smooth
    i_max = smooth(f1_curve.mean(0), 0.1).argmax()
    max_f1_conf = px[i_max]
    
    # What is r_curve inside res_tau evaluated at tau?
    idx_tau = int(np.argmin(np.abs(px - tau)))
    r_curve_at_tau = [float(res_tau.box.r_curve[c, idx_tau]) for c in range(len(res_tau.box.r_curve))]
    
    return {
        'ultra_box_r': ultra_box_r,
        'max_f1_conf': float(max_f1_conf),
        'r_curve_at_tau': r_curve_at_tau,
        'px_tau': float(px[idx_tau])
    }

if __name__ == '__main__':
    print("Testing YOLO11n seed 0 on test split at tau=0.25:")
    res = analyze_model_tau('models/yolo11n/baseline/seed_0/weights/best.pt', 'test', 0.25)
    print("ultra_box_r (res.box.r):", res['ultra_box_r'])
    print("max_f1_conf:", res['max_f1_conf'])
    print("r_curve inside res_tau at tau=0.25:", res['r_curve_at_tau'])
