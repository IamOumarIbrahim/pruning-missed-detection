# Waits for the crop retraining job, then runs validation + test evaluation.
# Usage (in a second window):  pwsh -File scripts\wait_and_eval.ps1
$ErrorActionPreference = 'Stop'
Set-Location "C:\Dev\repos\Public repos\research\pruning-missed-detections"

# 1. Wait for the retraining job (the python process running run_codesign_multiseed.py)
$job = Get-CimInstance Win32_Process -Filter "Name like 'python%'" |
    Where-Object { $_.CommandLine -match 'run_codesign_multiseed\.py' } |
    Select-Object -First 1
if ($job) {
    Write-Host "Waiting for retraining job (PID $($job.ProcessId)) to finish ..."
    Wait-Process -Id $job.ProcessId
    Write-Host "Job finished at $(Get-Date -Format 'HH:mm:ss')."
} else {
    Write-Host "No retraining job is running, checking results now."
}

# 2. Make sure all 6 crop runs (320 and 160, seeds 72/73/74) are COMPLETED
$check = @'
import sys
import pandas as pd
d = pd.read_csv('results/codesign_benchmark.csv')
ok = True
for res in ('320x320 Crop', '160x160 Crop'):
    sub = d[(d.model == 'YOLO11N') & (d.input_res == res) & (d.status == 'COMPLETED')]
    seeds = sorted(sub.seed.astype(int).tolist())
    print(res, 'completed seeds:', seeds)
    ok = ok and seeds == [72, 73, 74]
sys.exit(0 if ok else 1)
'@
$check | python -
if ($LASTEXITCODE -ne 0) {
    throw "Not all 6 crop runs are COMPLETED in results\codesign_benchmark.csv. The job may have crashed, so evaluation was not started."
}

# 3. Validation + test evaluation -> results\codesign_val_test.md (+ .csv)
python scripts\eval_codesign_val_test.py
if ($LASTEXITCODE -ne 0) { throw "Evaluation failed, see the error above." }
Write-Host "Done. Report: results\codesign_val_test.md"
