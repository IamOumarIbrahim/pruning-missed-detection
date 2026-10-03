import json
import numpy as np
import pandas as pd
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CLASS_NAMES = ['yawning', 'hand_over_mouth', 'drinking', 'phone_use']

def run_r4_analysis():
    print("=" * 80)
    print("MANDATE R4: EVENTS, SAMPLING, AND DATA PROVENANCE")
    print("=" * 80)

    # 1. Load Ground Truth and Events
    df_gt = pd.read_parquet(REPO_ROOT / 'results' / 'phase0' / 'ground_truth.parquet')
    df_events = pd.read_parquet(REPO_ROOT / 'results' / 'phase0' / 'events.parquet')

    print(f"Total events in dataset: {len(df_events)}")
    print(f"Events per split:\n{df_events['split'].value_counts()}")
    print(f"\nEvents per class:\n{df_events['class_name'].value_counts()}")

    # Subject x Class event count crosstab
    ct = pd.crosstab(df_events['subject'], df_events['class_name'])
    print("\nSubject x Class event matrix:")
    print(ct.to_string())

    # Save detailed event table with video IDs, start_frame, end_frame, num_frames
    df_events_sorted = df_events.sort_values(by=['subject', 'video', 'class_name', 'start_frame']).reset_index(drop=True)
    out_events_csv = REPO_ROOT / 'results' / 'phase0' / 'r4_events_full_table.csv'
    df_events_sorted.to_csv(out_events_csv, index=False)
    print(f"\nFull event table saved to: {out_events_csv}")

    # Inspect Subject 10 video structure
    s10 = df_events[df_events['subject'] == 'subject_10']
    print("\nSubject 10 Events across Videos:")
    print(s10[['event_id', 'video', 'class_name', 'start_frame', 'end_frame', 'num_frames']].to_string(index=False))

    # Video distribution by class across entire dataset
    vid_ct = pd.crosstab(df_events['video'], df_events['class_name'])
    print("\nClass presence by Video Number across entire dataset:")
    print(vid_ct.to_string())

if __name__ == '__main__':
    run_r4_analysis()
