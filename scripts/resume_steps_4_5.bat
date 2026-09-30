@echo off
cd /d "%~dp0\.."
echo ============================================================
echo  Resuming pipeline: Steps 4-5 (sequential)
echo ============================================================
echo.

call scripts\step4_joint_sweep.bat %*
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

call scripts\step5_selection.bat %*
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

echo.
echo ============================================================
echo  All steps complete.
echo ============================================================
