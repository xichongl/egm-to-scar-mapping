# Data Directory

This directory contains preprocessed EGM arrays and label arrays for model development, training, and evaluation.

## Directory Structure

```
data/
├── sample_data/                # Mock data for examples (provided)
│   ├── egm_signals.npy        # Synthetic EGM signals
│   ├── egm_labels.npy          # Scar tissue labels
│   └── README.md
└── README.md                   # This file
```

## Data Formats Specification

### EGM Signals: `egm_signals.npy`

**Shape:** `(n_samples, 3, 2500)` - NumPy array

**Dimensions:**
- `n_samples`: Number of recorded sites or data points
- `3`: Three channels
  - Channel 0: Unipolar EGM recording
  - Channel 1: Bipolar EGM recording
  - Channel 2: Reference channel
- `2500`: Time samples at 1000 Hz sampling rate (2.5 seconds)

**Data Type:** `float32` or `float64`

**Units:** Millivolts (mV)

**Range:** Typically -5 to +5 mV

**Preprocessing Expected:**
- Baseline wandering removed (high-pass filter)
- 50/60 Hz powerline noise removed
- No saturation
- Reasonable SNR (>10 dB)

**Example Creation:**
```python
import numpy as np

# Load your raw data
egm_signals = np.zeros((n_samples, 3, 2500), dtype=np.float32)

# ... load your data into this array ...

# Save for use with pipeline
np.save('data/your_data/egm_signals.npy', egm_signals)
```

### Labels: `egm_labels.npy`

**Shape:** `(n_samples, 3)` - NumPy array

**Dimensions:**
- `n_samples`: Must match egm_signals.npy
- `3`: Three scar classes (binary multi-label format)
  - Class 0: Endocardial scar (0 or 1)
  - Class 1: Mid-myocardial scar (0 or 1)
  - Class 2: Epicardial scar (0 or 1)

**Data Type:** `int32` or `uint8`

**Values:** 0 = absence, 1 = presence

**Notes:**
- A single measurement point can have scar in multiple layers (multi-label)
- Some samples may have no scar (all zeros)
- Labels are expected to be prepared upstream before using this repository

**Example Creation:**
```python
import numpy as np

# Ground truth labels prepared by your upstream labeling workflow
egm_labels = np.zeros((n_samples, 3), dtype=np.int32)

# ... assign labels based on anatomy ...

# Save
np.save('data/your_data/egm_labels.npy', egm_labels)
```

## Using Your Own Data

### Step 1: Prepare Data Files

Create a new directory for your data:
```bash
mkdir data/your_study_name
```

Add your data files:
- `egm_signals.npy` - EGM recordings
- `egm_labels.npy` - Scar tissue labels
- `patient_metadata.json` (optional) - Patient/recording info

### Step 2: Update Configuration

Edit `config.yaml` to point to your data:
```yaml
paths:
  data_dir: "data/your_study_name"   # Change from sample_data
  output_dir: "output"
  model_dir: "models"
```

### Step 3: Run Pipeline

```bash
# Start Jupyter
jupyter lab

# Open notebooks and update data loading cell:
data_dir = "data/your_study_name"  # Change from sample_data
```

## Data Loading in Code

### Option 1: Direct NumPy Loading
```python
import numpy as np

X = np.load("data/your_data/egm_signals.npy")
y = np.load("data/your_data/egm_labels.npy")

print(f"X shape: {X.shape}")  # Should be (n_samples, 3, 2500)
print(f"y shape: {y.shape}")  # Should be (n_samples, 3)
```

### Option 2: Using Project API
```python
from src import data_loading

# Load from assumed file structure
signals_dict, labels_dict = data_loading.load_egm_data("data/your_data")
X = signals_dict['signals']
y = labels_dict['labels']
```

## Data Validation

### Quick Checklist
- [ ] Both files present: `egm_signals.npy` and `egm_labels.npy`
- [ ] Correct shapes: `(n_samples, 3, 2500)` and `(n_samples, 3)`
- [ ] Correct data types: float and int
- [ ] No NaN or Inf values
- [ ] Signals in reasonable range: ±5 mV typical
- [ ] Label values only 0 or 1
- [ ] Both files have same n_samples

### Validation Script
```python
import numpy as np

def validate_data(signals_path, labels_path):
    """Validate data files."""
    X = np.load(signals_path)
    y = np.load(labels_path)
    
    errors = []
    
    # Check shapes
    if len(X.shape) != 3:
        errors.append(f"X should be 3D, got {X.shape}")
    if X.shape[1] != 3:
        errors.append(f"X should have 3 channels, got {X.shape[1]}")
    if X.shape[2] != 2500:
        errors.append(f"X should have 2500 timesteps, got {X.shape[2]}")
    
    if len(y.shape) != 2:
        errors.append(f"y should be 2D, got {y.shape}")
    if y.shape[1] != 3:
        errors.append(f"y should have 3 classes, got {y.shape[1]}")
    
    if X.shape[0] != y.shape[0]:
        errors.append(f"Sample count mismatch: {X.shape[0]} vs {y.shape[0]}")
    
    # Check values
    if np.any(np.isnan(X)):
        errors.append("X contains NaN values")
    if np.any(np.isnan(y)):
        errors.append("y contains NaN values")
    
    if not np.all((y == 0) | (y == 1)):
        errors.append("y contains values other than 0 or 1")
    
    if errors:
        print("Validation FAILED:")
        for error in errors:
            print(f"  - {error}")
        return False
    else:
        print("Validation PASSED ✓")
        print(f"  Signals shape: {X.shape}")
        print(f"  Labels shape: {y.shape}")
        print(f"  Signal range: [{X.min():.4f}, {X.max():.4f}]")
        print(f"  Label distribution: {np.sum(y, axis=0)}")
        return True

# Run validation
validate_data("data/your_data/egm_signals.npy", 
              "data/your_data/egm_labels.npy")
```

## Data Preprocessing Pipeline

The project includes built-in preprocessing:

```python
from src import data_processing

# 1. Bandpass filter (1-250 Hz)
X_filtered = data_processing.bandpass_filter(X, lowcut=1, highcut=250)

# 2. Normalize channels (peak normalization)
X_normalized = data_processing.normalize_channels(X_filtered, method="peak")

# 3. Generate spectrograms (for CNN-STFT)
X_spec = data_processing.compute_multiscale_spectrograms(X_normalized)

# 4. Stratified split
(X_train, y_train), (X_val, y_val), (X_test, y_test) = \
    data_processing.split_data_stratified(X_normalized, y)
```

## Privacy & Data Sharing

⚠️ **Important**: This repository does NOT include actual patient data due to privacy regulations (HIPAA, GDPR, etc.).

- Real patient data should be stored securely
- Use the mock data (`sample_data/`) for public repository
- Your private data belongs in `data/your_study_name/`
- Add `data/*/` to `.gitignore` (don't commit real data)

## Additional Resources

- [NumPy Data I/O](https://numpy.org/doc/stable/reference/routines.io.html)
- [HDF5 for larger datasets](https://www.h5py.org/)
- [MATLAB .mat files](https://docs.scipy.org/doc/scipy/reference/io.html#matlab)

---

**Questions?** Check the main [README.md](../README.md) or notebook examples in `notebooks/`.
