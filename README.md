# EGM to Scar Mapping: Model Development, Training, and Evaluation

A machine learning framework for model development, training, and evaluation on intracardiac electrograms (EGMs) using multi-scale spectrograms and self-supervised transformers.

## Project Overview

This repository contains tools for:
- **Data Loading & Processing**: Load preprocessed EGM arrays and prepare model-ready inputs
- **Neural Network Models**:
  - **CNN-STFT**: Multi-scale spectrogram-based CNN for supervised scar tissue prediction
  - **Transformer**: Self-supervised transformer with multiple masking strategies and contrastive learning
- **Evaluation**: Comprehensive metrics and visualization tools

## Key Features

✓ **Modular Design**: Clean separation of data loading, processing, model, and evaluation code  
✓ **Mock Data Support**: Realistic synthetic EGM data for testing without patient data  
✓ **TensorFlow/PyTorch Compatible**: Model specifications work with both frameworks  
✓ **Configurable**: YAML-based configuration for easy parameter tuning  
✓ **Reproducible**: Complete data pipeline with documented parameters  

## Project Structure

```
egm-to-scar-mapping/
├── README.md                       # This file
├── requirements.txt                # Python dependencies
├── config.yaml                     # Configuration parameters
├── generate_mock_data.py          # Generate synthetic data for testing
│
├── src/                            # Main source code
│   ├── __init__.py
│   ├── config.py                  # Configuration management
│   ├── data_loading.py            # Preprocessed dataset I/O utilities
│   ├── data_processing.py         # Signal processing & preprocessing
│   ├── utils.py                   # Helper functions
│   └── models/
│       ├── __init__.py
│       ├── cnn_stft.py           # CNN-STFT model architecture
│       └── transformer.py         # Transformer model architecture
│
├── notebooks/                      # Jupyter notebooks for workflows
│   ├── 00_generate_and_visualize_mock_data.ipynb # Generate and inspect mock data
│   ├── 01_load_preprocessed_inputs.ipynb # Load and inspect preprocessed model inputs
│   ├── 02_train_cnn_stft.ipynb     # Train CNN-STFT model
│   ├── 03_train_transformer.ipynb  # Train Transformer model
│   ├── 04_transformer_architecture.ipynb # Transformer architecture walkthrough
│   └── 05_evaluate_models.ipynb    # Model evaluation & visualization
│
└── data/
    ├── sample_data/               # Mock data for examples
    │   ├── egm_signals.npy
    │   ├── egm_labels.npy
    │   └── README.md
    └── README.md                  # Instructions for adding your data
```

## Quick Start

### 1. Installation

```bash
# Clone repository
git clone <repository-url>
cd egm-to-scar-mapping

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Mock Data

```bash
# Generate synthetic EGM signals and labels for testing
python generate_mock_data.py --output-dir data/sample_data --n-samples 100
```

### 3. Run Example Notebooks

```bash
# Start Jupyter
jupyter lab

# Open and run notebooks in order:
# 0. 00_generate_and_visualize_mock_data.ipynb - Generate and inspect mock data (optional)
# 1. 01_load_preprocessed_inputs.ipynb - Load and inspect preprocessed inputs
# 2. 02_train_cnn_stft.ipynb - Train CNN-STFT model
# 3. 03_train_transformer.ipynb - Train Transformer model
# 4. 04_transformer_architecture.ipynb - Review Transformer design
# 5. 05_evaluate_models.ipynb - Evaluate results
```

## Data Format Specification

### Input: EGM Signals

Expected shape: `(n_samples, 3_channels, 2500_timesteps)`

**Channels:**
- Channel 0: Unipolar EGM recording
- Channel 1: Bipolar EGM recording  
- Channel 2: Reference channel

**Sampling:** 1000 Hz (2.5 seconds of recording)

**Units:** Millivolts (mV)

### Output: Labels

Expected shape: `(n_samples, 3_classes)`

Binary multi-label format:
- **Class 0**: Endocardial scar tissue
- **Class 1**: Mid-myocardial scar tissue
- **Class 2**: Epicardial scar tissue

Values: 0 (absence) or 1 (presence)

These labels are assumed to be prepared externally before using this repository.

## Using Your Own Data

### 1. Prepare Data Files

Organize your data in `data/` directory with structure:
```
data/
├── sample_data/           # Mock data (provided)
├── your_data/             # Your patient data
│   ├── patient_001/
│   │   ├── egm_signals.npy      # Shape: (n_points, 3, 2500)
│   │   ├── egm_labels.npy       # Shape: (n_points, 3)
│   │   └── patient_metadata.json
│   └── patient_002/
│       └── ...
└── README.md
```

### 2. Update Configuration

Edit `config.yaml` to point to your data:
```yaml
paths:
  data_dir: "data/your_data"      # Changed from sample_data
  output_dir: "output"
  model_dir: "models"

model_params:
  cnn_stft:
    batch_size: 32
  transformer:
    batch_size: 32
```

### 3. Load and Run Pipeline

```python
from src import config, data_loading, data_processing

# Load configuration
cfg = config.get_config("config.yaml")

# Load your raw arrays
signals_dict, labels_dict = data_loading.load_egm_data(str(cfg.paths['data_dir']))
X = signals_dict['signals']
y = labels_dict['labels']

# Split into train/validation/test
(X_train, y_train), (X_val, y_val), (X_test, y_test) = data_processing.split_data_stratified(
  X, y, test_ratio=cfg.get_data_param('test_set_ratio', 0.2), val_ratio=cfg.get_data_param('val_set_ratio', 0.1)
)

# Process signals
X_train_filtered = data_processing.bandpass_filter(X_train, lowcut=1.0, highcut=250.0)
```

## Model Architectures

### CNN-STFT

**Purpose**: Supervised scar tissue classification using multi-scale spectrograms

**Input**: 128×128 multi-scale STFT spectrograms (3 scales × 3 channels = 9 channels)

**Architecture**:
- Conv2D → BatchNorm → ReLU
- 3× Residual blocks with max pooling
- Global average pooling
- 2× Dense layers
- Sigmoid output (binary multi-label)

**Training**: 10 epochs, batch size 32, learning rate 1e-5

### Transformer

**Purpose**: Self-supervised representation learning from unlabeled EGM data

**Input**: Raw EGM signals (n_channels=3, seq_length=2500)

**Architecture**:
- Patch embedding (50 timesteps/patch)
- 8-layer transformer encoder (d_model=256, nhead=8)
- Multiple reconstruction heads (cross-channel, intra-channel, autoregressive)
- Contrastive learning projection head

**Training**: 50 epochs with multiple masking strategies:
- Cross-channel masking (33%)
- Intra-channel masking (33%)
- Autoregressive masking (34%)

**Data Augmentation**: Gaussian noise, time shift, channel dropout

## Configuration Parameters

Edit `config.yaml` to customize:

```yaml
# Model hyperparameters
model_params:
  cnn_stft:
    batch_size: 32
    epochs: 10
    learning_rate: 1.0e-05
    stft_scales: 3
    spectrogram_size: 128

  transformer:
    batch_size: 32
    epochs: 50
    patch_size: 50
    d_model: 256
    nhead: 8
```

## API Reference

### Core Modules

#### `config.py`
- `Config`: Configuration class for managing paths and parameters
- `get_config()`: Get global config instance

#### `data_loading.py`
- `load_egm_data()`: Load EGM signals and labels
- `load_processed_datasets()`: Load train/val/test splits

#### `data_processing.py`
- `bandpass_filter()`: Apply Butterworth bandpass filter
- `normalize_channels()`: Peak or z-score normalization
- `compute_multiscale_spectrograms()`: Generate multi-scale STFT
- `split_data_stratified()`: Stratified train/val/test split
- `generate_synthetic_egm_data()`: Create synthetic signals for testing

#### `models/cnn_stft.py`
- `create_cnn_stft_model()`: CNN-STFT model specification
- `CNNSTFTHyperModel`: Hyperparameter tuning search space

#### `models/transformer.py`
- `create_egm_transformer_model()`: Transformer architecture spec
- `PretrainingDatasetSpec`: Masking and augmentation strategies
- `LossWeights`: Multi-task loss weight definitions

### Example Usage

```python
import numpy as np
from src import data_loading, data_processing, config
from src.models import cnn_stft

# Load configuration
cfg = config.get_config("config.yaml")

# Load data
signals = np.load("data/sample_data/egm_signals.npy")
labels = np.load("data/sample_data/egm_labels.npy")

# Normalize signals
normalized = data_processing.normalize_channels(signals, method="peak")

# Generate spectrograms
spectrograms = data_processing.compute_multiscale_spectrograms(
    normalized,
    frame_lengths=[64, 256, 512],
    output_size=128
)

# Split data
(X_train, y_train), (X_val, y_val), (X_test, y_test) = data_processing.split_data_stratified(
    spectrograms, labels, test_ratio=0.2, val_ratio=0.1
)

# Create model spec
model_spec = cnn_stft.create_cnn_stft_model(
    input_shape=(128, 128, 9),
    n_classes=3,
    initial_filters=16
)

print(model_spec)
```

## Citation

If you use this code in your research, please cite:

```bibtex
@software{egm_to_scar_2024,
  title={EGM to Scar Mapping: Machine Learning Framework for Cardiac Electrogram Analysis},
  author={Your Name and Team},
  year={2024},
  url={https://github.com/yourusername/egm-to-scar-mapping}
}
```

## License

MIT License - See LICENSE file for details

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## Troubleshooting

### Data Loading Issues

**Problem**: `FileNotFoundError: Signals file not found`

**Solution**: Ensure your data directory contains `egm_signals.npy` and `egm_labels.npy`

```python
import os
print(os.listdir("data/sample_data"))  # Check files exist
```

### Dimensional Mismatch

**Problem**: `ValueError: shapes (100, 2500) and (100, 2500, 3) not aligned`

**Solution**: Data should have shape `(n_samples, n_channels, n_timesteps)`

```python
signals = np.load("egm_signals.npy")
print(signals.shape)  # Should be (n_samples, 3, 2500)

# Transpose if needed
if signals.shape[1] != 3:
    signals = np.transpose(signals, (0, 2, 1))
```

## Support

For questions or issues:
1. Check existing GitHub issues
2. Review example notebooks in `notebooks/` directory
3. Check documentation in docstrings

## Acknowledgments

This research was supported by [funding sources/collaborators if applicable].

## Authors

Research Team

## Version

1.0.0 - Initial Release (2024)

---

**Latest Update**: March 2024  
**Status**: Active Development
