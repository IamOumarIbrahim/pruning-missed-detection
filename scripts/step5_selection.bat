@echo off
cd /d "%~dp0\.."
echo ============================================================
echo  Step 5: Final selection + table generation
echo ============================================================

python scripts\run_selection.py %*
if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: Selection failed with exit code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

echo.
python scripts\generate_tables.py
echo.
echo Step 5 complete.
