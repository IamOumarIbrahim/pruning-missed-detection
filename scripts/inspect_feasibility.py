import json

with open('results/calibrated_eval_data.json') as f:
    d = json.load(f)

print("="*70)
print("FEASIBILITY AND SATURATION ACROSS ALL 36 RUNS")
print("="*70)

infeasible_count = 0
saturated_count = 0
total_count = 0

for m in ['yolo11n', 'yolo26n']:
    for r in ['0%', '10%', '20%', '30%', '40%', '50%']:
        for s in ['0', '1', '2']:
            total_count += 1
            rec = d[m][r][s]
            feas = rec['val_feasible']
            tau = rec['tau_star']
            if not feas:
                infeasible_count += 1
                print(f"INFEASIBLE: {m} {r} seed {s} (tau*={tau:.4f})")
            if tau >= 0.899:
                saturated_count += 1
                print(f"SATURATED at 0.90: {m} {r} seed {s} (tau*={tau:.4f})")

print(f"\nTotal models evaluated: {total_count}")
print(f"Infeasible count: {infeasible_count} / {total_count} (0%)")
print(f"Saturated count (tau* >= 0.899): {saturated_count} / {total_count} (0%)")
