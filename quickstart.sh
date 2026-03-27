#!/usr/bin/env bash
# Quick Start Script for EGM-to-Scar-Mapping Repository

set -e  # Exit on error

echo "======================================================================"
echo "EGM-to-Scar-Mapping: Quick Start Setup"
echo "======================================================================"

# Step 1: Check Python
echo ""
echo "[1/5] Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 not found. Please install Python 3.8+"
    exit 1
fi
python3 --version

# Step 2: Create virtual environment
echo ""
echo "[2/5] Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✓ Virtual environment created: ./venv"
else
    echo "✓ Virtual environment already exists"
fi

# Step 3: Activate and upgrade pip
echo ""
echo "[3/5] Activating venv and installing dependencies..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

python3 -m pip install --upgrade pip setuptools wheel > /dev/null 2>&1
pip install -r requirements.txt > /dev/null 2>&1
echo "✓ Python dependencies installed"

# Step 4: Generate mock data
echo ""
echo "[4/5] Generating mock data..."
python3 generate_mock_data.py --n-samples 50 --output-dir data/sample_data
echo "✓ Mock data generated in: ./data/sample_data/"

# Step 5: Verify installation
echo ""
echo "[5/5] Verifying installation..."
python3 << 'EOF'
import sys
try:
    from src import config, data_loading, data_processing
    from src.models import cnn_stft, transformer
    import numpy as np
    import tensorflow as tf
    print("✓ All imports successful")
    print(f"  - Python version: {sys.version.split()[0]}")
    print(f"  - NumPy: {np.__version__}")
    print(f"  - TensorFlow: {tf.__version__}")
except Exception as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)
EOF

echo ""
echo "======================================================================"
echo "Setup Complete! 🎉"
echo "======================================================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Start Jupyter:"
echo "   jupyter lab"
echo ""
echo "2. Open notebooks in this order:"
echo "   - notebooks/00_generate_and_visualize_mock_data.ipynb (optional mock-data QA)"
echo "   - notebooks/01_load_preprocessed_inputs.ipynb  (preprocessed inputs)"
echo "   - notebooks/02_train_cnn_stft.ipynb             (CNN-STFT training)"
echo "   - notebooks/03_train_transformer.ipynb          (Transformer training)"
echo "   - notebooks/04_transformer_architecture.ipynb   (model details)"
echo "   - notebooks/05_evaluate_models.ipynb            (evaluation)"
echo ""
echo "3. To use your own data:"
echo "   - Create: data/your_study_name/"
echo "   - Copy: egm_signals.npy and egm_labels.npy"
echo "   - Edit: config.yaml - set data_dir path"
echo ""
echo "======================================================================"
