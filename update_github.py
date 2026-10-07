import subprocess
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
INTERVAL_SECONDS = 30 * 60  # 30 minutes


def run_command(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )


def sync():
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] Checking for updates...", flush=True)

    # git add .
    add_res = run_command(["git", "add", "."])
    if add_res.returncode != 0:
        print(f"[{now}] Error running git add: {add_res.stderr.strip()}", flush=True)
        return

    # Check status to see if there is anything to commit
    status_res = run_command(["git", "status", "--porcelain"])
    if status_res.stdout.strip():
        # git commit -m "update"
        commit_res = run_command(["git", "commit", "-m", "update"])
        print(f"[{now}] git commit: {commit_res.stdout.strip()}", flush=True)
        if commit_res.returncode != 0 and commit_res.stderr.strip():
            print(f"[{now}] Commit error: {commit_res.stderr.strip()}", flush=True)
    else:
        print(f"[{now}] No changes to commit.", flush=True)

    # git push
    push_res = run_command(["git", "push"])
    if push_res.returncode == 0:
        push_output = push_res.stdout.strip() or push_res.stderr.strip()
        print(f"[{now}] git push successful: {push_output}", flush=True)
    else:
        print(f"[{now}] git push failed: {push_res.stderr.strip()}", flush=True)


def main():
    print(f"Starting GitHub auto-updater in {REPO_ROOT}", flush=True)
    print(f"Interval: every 30 minutes ({INTERVAL_SECONDS} seconds)", flush=True)
    while True:
        try:
            sync()
        except Exception as e:
            print(f"Unexpected error: {e}", flush=True)

        print(f"Sleeping for 30 minutes...\n", flush=True)
        time.sleep(INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
