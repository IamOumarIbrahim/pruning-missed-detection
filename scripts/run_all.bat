@echo off
cd /d "%~dp0\.."
echo ============================================================
echo  Full pipeline: Steps 1-5 (sequential)
echo ============================================================
echo.

call scripts\step1_baseline.bat %*
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

call scripts\step2_pruning_sweep.bat %*
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

call scripts\step3_quant_ablation.bat %*
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

call scripts\step4_joint_sweep.bat %*
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

call scripts\step5_selection.bat %*
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

echo.
echo ============================================================
echo  All steps complete.
echo ============================================================
pause
