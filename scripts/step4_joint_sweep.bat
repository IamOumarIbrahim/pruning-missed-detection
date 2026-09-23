@echo off
cd /d "%~dp0\.."
echo ============================================================
echo  Step 4: Joint pruning x INT8 + conditional QAT
echo ============================================================

echo --- Step 4a: PTQ INT8 at all pruning ratios ---
python scripts\run_joint_sweep.py %*
if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: Step 4a failed with exit code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo --- Step 4b: QAT where PTQ-INT8 fails ---
python scripts\run_qat.py %*
if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: Step 4b QAT failed with exit code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo Step 4 complete.
