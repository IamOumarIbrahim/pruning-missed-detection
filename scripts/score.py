"""Turn cached detections into results/tables/runs.csv (seconds, no model needed)."""
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dms import evaluate, runs  # noqa: E402
from dms.paths import DETECTIONS, TABLES, enter_root  # noqa: E402


def main():
    enter_root()
    TABLES.mkdir(parents=True, exist_ok=True)
    groups = defaultdict(list)
    for run in runs.phase1() + runs.phase2() + runs.phase3():
        if all((DETECTIONS / run.id / f"{s}.npz").exists() for s in ("val", "test")):
            groups[run.id.rsplit("/", 1)[0]].append(run)
    rows = []
    for name, members in groups.items():
        scored = {r: evaluate.scored_run(r) for r in members}
        tau = evaluate.select_config_tau([s["val"] for s in scored.values()])
        rows += [evaluate.row(r, s, tau) for r, s in scored.items()]
        print(f"{name}: {len(members)} seeds, tau={tau:.3f}", flush=True)
    with open(TABLES / "runs.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
