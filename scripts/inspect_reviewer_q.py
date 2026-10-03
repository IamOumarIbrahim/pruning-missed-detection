import json

with open('results/calibrated_eval_data.json') as f:
    d = json.load(f)

print("="*70)
print("QUESTION 2: BASELINE CONTROL (0% PRUNING, ALL SEEDS, BOTH MODELS)")
print("="*70)

for m in ['yolo11n', 'yolo26n']:
    print(f"\n--- {m.upper()} BASELINE (0% Pruning) ---")
    cal = d['calibration_info'][m]
    print(f"Val R_base mean (tau=0.25): {cal['r_base_val']:.4f} | Val R_floor mean: {cal['r_floor']:.4f}")
    
    test_passed_count = 0
    for s in ['0', '1', '2']:
        rec = d[m]['0%'][s]
        vf = rec['val_r_floor']
        tau = rec['tau_star']
        val_rec_star = rec['val_min_recall_at_tau_star']
        t_star = rec['test_at_tau_star']
        t_25 = rec['test_at_tau_25']
        passed = t_star['passed_floor']
        if passed:
            test_passed_count += 1
            
        print(f"\nSeed {s}:")
        print(f"  Val Floor (R_base_seed - 0.05): {vf:.4f}")
        print(f"  Val tau*:                       {tau:.4f}")
        print(f"  Val Min Recall @ tau*:          {val_rec_star:.4f}")
        print(f"  Test Min Recall @ tau=0.25:     {t_25['min_recall']:.4f} (worst: {t_25['worst_class']})")
        print(f"  Test Min Recall @ tau*:         {t_star['min_recall']:.4f} (worst: {t_star['worst_class']})")
        print(f"  Test Margin vs Val Floor:       {t_star['achieved_margin']*100:+.2f} pp")
        print(f"  Passed Val Floor on Test?       {passed}")
        print(f"  Per-class Recall on Test @ tau*:")
        for cname, rval in t_star['per_class_recall'].items():
            print(f"    - {cname:15s}: {rval:.4f}")
        print(f"  Per-class Precision on Test @ tau*:")
        for cname, pval in t_star['per_class_precision'].items():
            print(f"    - {cname:15s}: {pval:.4f}")
            
    print(f"\nSummary for {m} 0%: Passed Test Floor: {test_passed_count}/3 ({test_passed_count/3*100:.1f}%)")
