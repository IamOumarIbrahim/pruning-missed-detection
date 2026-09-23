@echo off
cd /d "%~dp0\.."
echo ============================================================
echo  Step 2: Pruning sweep at FP32 (R%% - 5R%%)
echo ============================================================
python scripts\run_pruning_sweep.py %*
if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: Step 2 failed with exit code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)
echo.
echo Step 2 complete.
