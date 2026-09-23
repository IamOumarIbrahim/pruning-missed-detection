@echo off
cd /d "%~dp0\.."
echo ============================================================
echo  Step 3: Quantization-only ablation (FP16 / INT8 / INT4)
echo ============================================================
python scripts\run_quant_ablation.py %*
if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: Step 3 failed with exit code %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)
echo.
echo Step 3 complete.
