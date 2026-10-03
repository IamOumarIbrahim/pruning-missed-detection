import json
import numpy as np
from pathlib import Path

# Verify coordinate matching between YOLO GT and predictions.json
def test_matching():
    # 1. Load sample prediction from test1
    p_path = Path("runs/detect/runs/test_cache_eval/test1/predictions.json")
    if not p_path.exists():
        print("test1 predictions.json not found, skipping direct verification")
        return
        
    with open(p_path) as f:
        preds = json.load(f)
        
    print(f"Loaded {len(preds)} predictions")
    img_id = preds[0]['image_id'] # e.g. subject_05_video_01_frame_0005
    parts = img_id.split('_')
    sub = f"{parts[0]}_{parts[1]}"
    vid = f"{parts[2]}_{parts[3]}"
    frame = f"{parts[4]}_{parts[5]}"
    lbl_file = Path(f"data/processed/RGB/labels/{sub}/{vid}/{img_id}.txt")
    print(f"Checking label file: {lbl_file}, exists: {lbl_file.exists()}")
    if lbl_file.exists():
        with open(lbl_file) as f:
            for line in f:
                print("  GT raw:", line.strip())

if __name__ == '__main__':
    test_matching()
