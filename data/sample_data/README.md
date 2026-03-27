# Mock EGM Data

This directory contains synthetic/mock EGM data for testing and development.

## Files

- `egm_signals.npy`: Synthetic EGM signals
  - Shape: (n_samples, 3 channels, 2500 timesteps)
  - Channels: Unipolar, Bipolar, Reference
  - Sampling rate: 1000 Hz
  
- `egm_labels.npy`: Binary scar tissue labels
  - Shape: (n_samples, 3 classes)
  - Classes: Endocardial, Mid-myocardial, Epicardial

## Usage

```python
import numpy as np

# Load signals and labels
signals = np.load('egm_signals.npy')
labels = np.load('egm_labels.npy')

print(f"Signals shape: {signals.shape}")
print(f"Labels shape: {labels.shape}")
```

## Data Characteristics

The synthetic signals include:
- ECG-like morphology with P wave, QRS complex, and T wave
- Beat-to-beat heart rate variability
- Baseline drift and respiration-like wander
- Realistic noise levels
- 50/60 Hz powerline contamination

Labels follow realistic scar distribution:
- ~70% of samples have no scar
- ~30% of samples have scar in one or more layers
