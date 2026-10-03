import os
from pathlib import Path
from collections import defaultdict

REPO_ROOT = Path('.')
DATA_RGB = REPO_ROOT / 'data' / 'processed' / 'RGB'

splits = {
    'train': DATA_RGB / 'yolo' / 'train.txt',
    'val': DATA_RGB / 'yolo' / 'val.txt',
    'test': DATA_RGB / 'yolo' / 'test.txt'
}

class_names = ['yawning', 'hand_over_mouth', 'drinking', 'phone_use']

print("="*70)
print("QUESTION 3: DATASET COUNTS PER SUBJECT, VIDEO, CLASS, AND SPLIT")
print("="*70)

# Track per split: subjects, videos, frames, class_counts
for split_name, txt_file in splits.items():
    print(f"\n--- SPLIT: {split_name.upper()} ---")
    with open(txt_file) as f:
        img_lines = [l.strip() for l in f if l.strip()]
        
    subject_frames = defaultdict(int)
    subject_videos = defaultdict(set)
    subject_class_counts = defaultdict(lambda: defaultdict(int))
    total_class_counts = defaultdict(int)
    
    for l in img_lines:
        # e.g., ./../images/subject_08/video_02/subject_08_video_02_frame_0355.jpg
        # label path: data/processed/RGB/labels/subject_08/video_02/subject_08_video_02_frame_0355.txt
        parts = Path(l).parts
        # find subject_XX and video_YY
        sub = [p for p in parts if p.startswith('subject_')][0]
        vid = [p for p in parts if p.startswith('video_')][0]
        
        subject_frames[sub] += 1
        subject_videos[sub].add(vid)
        
        lbl_file = DATA_RGB / 'labels' / sub / vid / f"{Path(l).stem}.txt"
        if lbl_file.exists():
            with open(lbl_file) as lf:
                for line in lf:
                    lp = line.strip().split()
                    if lp:
                        c_id = int(lp[0])
                        c_name = class_names[c_id]
                        subject_class_counts[sub][c_name] += 1
                        total_class_counts[c_name] += 1

    print(f"Total Frames: {len(img_lines)}")
    print(f"Total Class Counts across split: {dict(total_class_counts)}")
    print("\nPer-Subject Breakdown:")
    for sub in sorted(subject_frames.keys()):
        vids = sorted(list(subject_videos[sub]))
        counts_str = ", ".join([f"{cn}: {subject_class_counts[sub][cn]}" for cn in class_names])
        print(f"  {sub} ({len(vids)} videos: {', '.join(vids)}): {subject_frames[sub]} frames | GT: {counts_str}")
