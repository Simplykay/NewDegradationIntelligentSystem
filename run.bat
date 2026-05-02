@echo off
REM Cotton Seed Quality Intelligence System — Local Launcher
REM Usage: run.bat [train|api|dashboard|all]

SET ROOT=%~dp0
SET VENV=%ROOT%.venv\Scripts\

if "%1"=="train" goto train
if "%1"=="api" goto api
if "%1"=="dashboard" goto dashboard

echo.
echo Cotton Seed Quality Intelligence System
echo ========================================
echo.
echo Commands:
echo   run.bat train      - Train all 6 models (first time, ~10-20 min)
echo   run.bat api        - Start FastAPI backend  (localhost:8000)
echo   run.bat dashboard  - Start Streamlit dashboard (localhost:8501)
echo.
echo Quick start:
echo   1. run.bat train
echo   2. In a new terminal: run.bat api
echo   3. In a new terminal: run.bat dashboard
echo   4. Open http://localhost:8501
echo.
goto end

:train
echo Training all 6 models...
cd /d "%ROOT%"
"%VENV%python.exe" train_models.py
goto end

:api
echo Starting FastAPI backend on http://localhost:8000 ...
cd /d "%ROOT%"
"%VENV%uvicorn.exe" app.api.main:app --host 0.0.0.0 --port 8000 --reload
goto end

:dashboard
echo Starting Streamlit dashboard on http://localhost:8501 ...
cd /d "%ROOT%\app\dashboard"
"%VENV%streamlit.exe" run app.py
goto end

:end
