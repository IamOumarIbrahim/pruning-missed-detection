import json

with open("results/table1_data.json") as f:
    d = json.load(f)

for r in d["raw_runs"]:
    if r["model"] == "yolo26n" and r["pruning_ratio"] in ["0%", "10%", "20%"]:
        print(f"Ratio {r['pruning_ratio']} Seed {r['seed']}:")
        print("  tau_star:", round(r["tau_star"], 3), "worst_class:", r["tau_star_eval"]["worst_class"], "min_recall:", round(r["tau_star_eval"]["min_recall"], 3))
        print("  per_class_recalls (tau_star):", {k: round(v, 3) for k, v in r["tau_star_eval"]["per_class_recall"].items()})
        print("  tau_25:", round(r["tau_25_eval"]["min_recall"], 3), "worst_class:", r["tau_25_eval"]["worst_class"])
        print("  per_class_recalls (tau_25):", {k: round(v, 3) for k, v in r["tau_25_eval"]["per_class_recall"].items()})
