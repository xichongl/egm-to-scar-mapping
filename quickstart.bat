@echo off
REM Quick Start Script for EGM-to-Scar-Mapping Repository (Windows)

echo ======================================================================
echo EGM-to-Scar-Mapping: Quick Start Setup
echo ======================================================================

REM Step 1: Check Python
echo.
echo [1/5] Checking Python installation...
python --version
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.8+
    pause
    exit /b 1
)

REM Step 2: Create virtual environment
echo.
echo [2/5] Creating virtual environment...
if not exist "venv" (
    python -m venv venv
    echo. Virtual environment created: .\venv
) else (
    echo. Virtual environment already exists
)

REM Step 3: Activate and upgrade pip
echo.
echo [3/5] Activating venv and installing dependencies...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip setuptools wheel > nul 2>&1
pip install -r requirements.txt > nul 2>&1
echo. Python dependencies installed

REM Step 4: Generate mock data
echo.
echo [4/5] Generating mock data...
python generate_mock_data.py --n-samples 50 --output-dir data/sample_data
echo. Mock data generated in: .\data\sample_data\

REM Step 5: Verify installation
echo.
echo [5/5] Verifying installation...
python -c "from src import config, data_loading, data_processing; from src.models import cnn_stft, transformer; print('✓ All imports successful')"
if errorlevel 1 (
    echo. ERROR: Import verification failed
    pause
    exit /b 1
)

echo.
echo ======================================================================
echo Setup Complete!
echo ======================================================================
echo.
echo Next steps:
echo.
echo 1. Start Jupyter:
echo    jupyter lab
echo.
echo 2. Open notebooks in this order:
echo    - notebooks/00_generate_and_visualize_mock_data.ipynb (optional mock-data QA)
echo    - notebooks/01_load_preprocessed_inputs.ipynb
echo    - notebooks/02_train_cnn_stft.ipynb
echo    - notebooks/03_train_transformer.ipynb
echo    - notebooks/04_transformer_architecture.ipynb
echo    - notebooks/05_evaluate_models.ipynb
echo.
echo 3. To use your own data:
echo    - Create: data/your_study_name/
echo    - Copy: egm_signals.npy and egm_labels.npy
echo    - Edit: config.yaml - set data_dir path
echo.
echo ======================================================================

pause
