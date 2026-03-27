#!/usr/bin/env python3
"""
Generate synthetic/mock data for testing and examples.

This script creates realistic synthetic EGM data and labels that can be used
for testing the pipeline without requiring actual patient data.

Usage:
    python generate_mock_data.py --output-dir data/sample_data --n-samples 100
"""

import argparse
import os
import sys
import numpy as np
from pathlib import Path


def generate_mock_egm_signals(n_samples: int = 100,
                              n_channels: int = 3,
                              n_timesteps: int = 2500,
                              sampling_rate: float = 1000.0,
                              random_state: int = 42) -> np.ndarray:
    """
    Generate synthetic EGM signals with realistic characteristics.
    
    Args:
        n_samples: Number of signals to generate.
        n_channels: Number of channels (unipolar, bipolar, reference).
        n_timesteps: Samples per signal.
        sampling_rate: Sampling rate in Hz.
        random_state: Random seed for reproducibility.
    
    Returns:
        Array of shape (n_samples, n_channels, n_timesteps).
    """
    np.random.seed(random_state)

    def gaussian_pulse(x: np.ndarray, center: float, width: float, amp: float) -> np.ndarray:
        """Simple Gaussian pulse used to compose ECG morphology."""
        return amp * np.exp(-0.5 * ((x - center) / width) ** 2)

    egm_signals = np.zeros((n_samples, n_channels, n_timesteps), dtype=np.float32)
    duration = n_timesteps / sampling_rate
    t = np.arange(n_timesteps, dtype=np.float32) / sampling_rate

    print(f"Generating {n_samples} synthetic EGM signals...")
    for sample in range(n_samples):
        # Per-sample heart rate with mild beat-to-beat variability
        bpm = np.random.uniform(55, 95)
        rr_mean = 60.0 / bpm
        beat_times = []
        beat_t = np.random.uniform(0.15, 0.35)
        while beat_t < duration + rr_mean:
            beat_times.append(beat_t)
            beat_t += rr_mean + np.random.normal(0.0, 0.04)

        # Channel-specific scaling to emulate unipolar, bipolar, and reference leads
        channel_profiles = [
            {"p": 1.00, "qrs": 1.00, "t": 1.00},
            {"p": 0.70, "qrs": 1.25, "t": 0.70},
            {"p": 0.50, "qrs": 0.85, "t": 0.55},
        ]

        for channel in range(n_channels):
            profile = channel_profiles[min(channel, len(channel_profiles) - 1)]
            signal = np.zeros(n_timesteps, dtype=np.float32)

            # Build morphology beat-by-beat: P wave, QRS complex, and T wave.
            for r_peak in beat_times:
                # Small channel-dependent timing offset (milliseconds)
                lead_delay = np.random.normal(0.0, 0.004)
                r_loc = r_peak + lead_delay

                p_amp = profile["p"] * np.random.uniform(0.06, 0.14)
                q_amp = -profile["qrs"] * np.random.uniform(0.10, 0.20)
                r_amp = profile["qrs"] * np.random.uniform(0.70, 1.30)
                s_amp = -profile["qrs"] * np.random.uniform(0.15, 0.35)
                t_amp = profile["t"] * np.random.uniform(0.18, 0.36)

                # Centers relative to R peak (seconds)
                p_loc = r_loc - np.random.uniform(0.18, 0.24)
                q_loc = r_loc - np.random.uniform(0.025, 0.045)
                s_loc = r_loc + np.random.uniform(0.025, 0.045)
                t_loc = r_loc + np.random.uniform(0.22, 0.34)

                # Widths (standard deviation in seconds)
                p_w = np.random.uniform(0.025, 0.045)
                q_w = np.random.uniform(0.006, 0.012)
                r_w = np.random.uniform(0.008, 0.014)
                s_w = np.random.uniform(0.008, 0.016)
                t_w = np.random.uniform(0.045, 0.085)

                signal += gaussian_pulse(t, p_loc, p_w, p_amp)
                signal += gaussian_pulse(t, q_loc, q_w, q_amp)
                signal += gaussian_pulse(t, r_loc, r_w, r_amp)
                signal += gaussian_pulse(t, s_loc, s_w, s_amp)
                signal += gaussian_pulse(t, t_loc, t_w, t_amp)

            # Baseline wander and respiration-like drift
            drift_f = np.random.uniform(0.12, 0.35)
            resp_f = np.random.uniform(0.18, 0.32)
            baseline = (
                np.random.uniform(0.04, 0.10) * np.sin(2 * np.pi * drift_f * t)
                + np.random.uniform(0.02, 0.06) * np.sin(2 * np.pi * resp_f * t + np.random.uniform(0, 2 * np.pi))
            )

            # Mild high-frequency and line-noise contamination
            white_noise = np.random.normal(0.0, np.random.uniform(0.008, 0.020), size=n_timesteps)
            line_noise = np.random.uniform(0.005, 0.02) * np.sin(
                2 * np.pi * 60.0 * t + np.random.uniform(0, 2 * np.pi)
            )

            signal = signal + baseline + white_noise + line_noise

            # Normalize to a stable dynamic range used by downstream notebooks
            signal_std = np.std(signal)
            if signal_std > 1e-8:
                signal = signal / signal_std * 0.5

            egm_signals[sample, channel, :] = signal
        
        if (sample + 1) % 20 == 0:
            print(f"  Generated {sample + 1}/{n_samples} samples")
    
    print(f"EGM signals shape: {egm_signals.shape}")
    print(f"Signal range: [{egm_signals.min():.4f}, {egm_signals.max():.4f}]")
    
    return egm_signals


def generate_mock_labels(n_samples: int = 100,
                         n_classes: int = 3,
                         random_state: int = 42) -> np.ndarray:
    """
    Generate synthetic scar tissue labels.
    
    Binary multi-label format where each class indicates presence of scar
    in that anatomical layer:
    - Class 0: Endocardial scar
    - Class 1: Mid-myocardial scar
    - Class 2: Epicardial scar
    
    Args:
        n_samples: Number of samples.
        n_classes: Number of label classes.
        random_state: Random seed.
    
    Returns:
        Binary label array of shape (n_samples, n_classes).
    """
    np.random.seed(random_state)
    
    labels = np.zeros((n_samples, n_classes), dtype=int)
    
    print(f"Generating {n_samples} synthetic labels ({n_classes} classes)...")
    
    # Create realistic label distribution
    # Some samples have no scar, some have single layer, some have multiple layers
    for sample in range(n_samples):
        # Probability of having scar: ~30%
        if np.random.rand() < 0.3:
            # Choose 1-3 layers
            n_layers = np.random.randint(1, n_classes + 1)
            chosen_classes = np.random.choice(n_classes, n_layers, replace=False)
            labels[sample, chosen_classes] = 1
    
    # Print statistics
    label_counts = np.sum(labels, axis=0)
    samples_with_scar = np.sum(np.sum(labels, axis=1) > 0)
    
    print(f"  Endocardial scar: {label_counts[0]} samples ({100*label_counts[0]/n_samples:.1f}%)")
    print(f"  Mid-myocardial scar: {label_counts[1]} samples ({100*label_counts[1]/n_samples:.1f}%)")
    print(f"  Epicardial scar: {label_counts[2]} samples ({100*label_counts[2]/n_samples:.1f}%)")
    print(f"  Samples with any scar: {samples_with_scar} ({100*samples_with_scar/n_samples:.1f}%)")
    print(f"  Labels shape: {labels.shape}")
    
    return labels


def save_mock_data(output_dir: str,
                   egm_signals: np.ndarray,
                   labels: np.ndarray) -> None:
    """
    Save mock data to files.
    
    Args:
        output_dir: Directory where to save files.
        egm_signals: EGM signal array.
        labels: Label array.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save raw data
    signals_path = os.path.join(output_dir, "egm_signals.npy")
    labels_path = os.path.join(output_dir, "egm_labels.npy")
    
    np.save(signals_path, egm_signals)
    np.save(labels_path, labels)
    
    print(f"\nSaved EGM signals to: {signals_path}")
    print(f"Saved labels to: {labels_path}")
    
    # Also create a simple README
    readme_path = os.path.join(output_dir, "README.md")
    readme_content = """# Mock EGM Data

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
"""
    
    with open(readme_path, "w") as f:
        f.write(readme_content)
    
    print(f"Saved README to: {readme_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic EGM data for testing."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/sample_data",
        help="Output directory for generated data (default: data/sample_data)"
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=100,
        help="Number of samples to generate (default: 100)"
    )
    parser.add_argument(
        "--n-channels",
        type=int,
        default=3,
        help="Number of EGM channels (default: 3)"
    )
    parser.add_argument(
        "--n-timesteps",
        type=int,
        default=2500,
        help="Timesteps per signal (default: 2500 = 2.5s at 1kHz)"
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("SYNTHETIC EGM DATA GENERATION")
    print("="*60)
    
    # Generate data
    egm_signals = generate_mock_egm_signals(
        n_samples=args.n_samples,
        n_channels=args.n_channels,
        n_timesteps=args.n_timesteps,
        random_state=args.random_seed
    )
    
    labels = generate_mock_labels(
        n_samples=args.n_samples,
        n_classes=3,
        random_state=args.random_seed
    )
    
    # Save data
    save_mock_data(args.output_dir, egm_signals, labels)
    
    print("\n" + "="*60)
    print("GENERATION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
