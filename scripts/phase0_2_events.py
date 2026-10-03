import pandas as pd
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
df_gt = pd.read_parquet(REPO_ROOT / 'results' / 'phase0' / 'ground_truth.parquet')

CLASS_NAMES = ['yawning', 'hand_over_mouth', 'drinking', 'phone_use']

print("="*75)
print("PHASE 0.2: EVENT SEGMENTATION AUDIT (GAP <= 5 FRAMES)")
print("="*75)

# Group by (split, subject, video, class_name) and cluster into events
event_records = []
event_counter = 0

for (split, sub, vid, cname), grp in df_gt.groupby(['split', 'subject', 'video', 'class_name']):
    grp_sorted = grp.sort_values('frame_num')
    frame_nums = grp_sorted['frame_num'].tolist()
    
    # Cluster consecutive frames with gap <= 5
    current_event = [frame_nums[0]]
    for f in frame_nums[1:]:
        if f - current_event[-1] <= 5:
            current_event.append(f)
        else:
            # Save event
            event_records.append({
                'event_id': f"evt_{event_counter:04d}",
                'split': split,
                'subject': sub,
                'video': vid,
                'class_name': cname,
                'start_frame': current_event[0],
                'end_frame': current_event[-1],
                'num_frames': len(current_event),
                'duration_span': current_event[-1] - current_event[0] + 1
            })
            event_counter += 1
            current_event = [f]
            
    # Save final event
    event_records.append({
        'event_id': f"evt_{event_counter:04d}",
        'split': split,
        'subject': sub,
        'video': vid,
        'class_name': cname,
        'start_frame': current_event[0],
        'end_frame': current_event[-1],
        'num_frames': len(current_event),
        'duration_span': current_event[-1] - current_event[0] + 1
    })
    event_counter += 1

df_events = pd.DataFrame(event_records)
df_events.to_parquet(REPO_ROOT / 'results' / 'phase0' / 'events.parquet', index=False)

print(f"Total Events Segmented: {len(df_events)}")

# Tabulate events and frames per class x subject x split
for split in ['train', 'val', 'test']:
    print(f"\n=======================================================")
    print(f"SPLIT: {split.upper()}")
    print(f"=======================================================")
    split_ev = df_events[df_events['split'] == split]
    split_gt = df_gt[df_gt['split'] == split]
    
    # Pivot table: Subject x Class (Events / Frames)
    subjects = sorted(split_ev['subject'].unique())
    
    print(f"{'Subject':12s} | " + " | ".join([f"{cn:16s}" for cn in CLASS_NAMES]))
    print("-" * 85)
    for sub in subjects:
        row_str = f"{sub:12s} | "
        for cn in CLASS_NAMES:
            n_ev = len(split_ev[(split_ev['subject'] == sub) & (split_ev['class_name'] == cn)])
            n_fr = len(split_gt[(split_gt['subject'] == sub) & (split_gt['class_name'] == cn)])
            row_str += f"{n_ev:2d} ev ({n_fr:3d} fr) | "
        print(row_str)
        
    print("-" * 85)
    tot_row = f"{'TOTAL':12s} | "
    smallest_event_count = 999999
    smallest_class = None
    for cn in CLASS_NAMES:
        n_ev = len(split_ev[split_ev['class_name'] == cn])
        n_fr = len(split_gt[split_gt['class_name'] == cn])
        if n_ev < smallest_event_count:
            smallest_event_count = n_ev
            smallest_class = cn
        tot_row += f"{n_ev:2d} ev ({n_fr:3d} fr) | "
    print(tot_row)
    print(f"\n>> SMALLEST EVENT COUNT in {split.upper()}: {smallest_class} with ONLY {smallest_event_count} EVENTS!")

