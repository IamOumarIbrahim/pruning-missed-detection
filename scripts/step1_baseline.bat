@echo off
cd /d "%~dp0\.."
echo ============================================================
echo  Step 1: Train baselines (YOLO11n + YOLO26n, K=3 seeds)
echo ============================================================
python scripts\train_baseline.py %*
if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: Step 1 failed with exit code %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)
echo.
echo Step 1 complete.
