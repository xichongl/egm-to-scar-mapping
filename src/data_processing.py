"""
Data processing utilities for EGM signals and model-ready representations.

This module provides preprocessing functions for:
- Signal filtering and normalization
- STFT spectrogram generation
- Data splitting and stratification
"""

import numpy as np
from typing import Tuple, List, Dict, Optional
from scipy import signal
from sklearn.model_selection import train_test_split
from collections import Counter


def normalize_channels(egm_signals: np.ndarray, method: str = "peak") -> np.ndarray:
    """
    Normalize EGM signals per channel.
    
    Args:
        egm_signals: Array of shape (n_samples, n_channels, n_timesteps).
        method: Normalization method - "peak" (default) or "std".
    
    Returns:
        Normalized signals with same shape as input.
    """
    if method == "peak":
        # Normalize each channel to peak absolute value of 1
        normalized = egm_signals.copy()
        for i in range(egm_signals.shape[1]):  # per channel
            max_val = np.max(np.abs(egm_signals[:, i, :]))
            if max_val > 0:
                normalized[:, i, :] = egm_signals[:, i, :] / max_val
    elif method == "std":
        # Normalize each channel to unit variance
        normalized = egm_signals.copy()
        for i in range(egm_signals.shape[1]):  # per channel
            std_val = np.std(egm_signals[:, i, :])
            if std_val > 0:
                normalized[:, i, :] = egm_signals[:, i, :] / std_val
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized


def bandpass_filter(egm_signals: np.ndarray, 
                    lowcut: float = 1.0, 
                    highcut: float = 250.0,
                    sampling_rate: float = 1000.0,
                    order: int = 4) -> np.ndarray:
    """
    Apply bandpass Butterworth filter to EGM signals.
    
    Args:
        egm_signals: Array of shape (n_samples, n_channels, n_timesteps).
        lowcut: Low cutoff frequency in Hz.
        highcut: High cutoff frequency in Hz.
        sampling_rate: Sampling rate in Hz.
        order: Filter order.
    
    Returns:
        Filtered signals with same shape as input.
    """
    nyquist = sampling_rate / 2.0
    low = lowcut / nyquist
    high = highcut / nyquist
    
    # Ensure valid frequency band
    low = np.clip(low, 0.001, 0.999)
    high = np.clip(high, low + 0.001, 0.999)
    
    b, a = signal.butter(order, [low, high], btype='band')
    
    filtered = np.zeros_like(egm_signals)
    for i in range(egm_signals.shape[1]):  # per channel
        filtered[:, i, :] = signal.filtfilt(b, a, egm_signals[:, i, :], axis=1)
    
    return filtered


def compute_stft_spectrogram(signal_data: np.ndarray,
                            frame_length: int = 256,
                            frame_step: int = 128,
                            output_size: int = 128) -> np.ndarray:
    """
    Compute STFT spectrogram for a single signal.
    
    Args:
        signal_data: 1D signal array.
        frame_length: STFT frame length.
        frame_step: STFT frame step.
        output_size: Resize spectrogram to (output_size, output_size).
    
    Returns:
        Log-magnitude spectrogram of shape (output_size, output_size).
    """
    # Compute STFT
    f, t, Zxx = signal.stft(signal_data, nperseg=frame_length, noverlap=frame_length-frame_step)
    
    # Take log magnitude
    spec = np.log(np.abs(Zxx) + 1e-10)
    
    # Resize to target size
    from scipy.ndimage import zoom
    current_shape = spec.shape
    scale_factors = (output_size / current_shape[0], output_size / current_shape[1])
    resized_spec = zoom(spec, scale_factors, order=1)
    
    return resized_spec


def compute_multiscale_spectrograms(egm_signals: np.ndarray,
                                    frame_lengths: List[int] = None,
                                    frame_steps: List[int] = None,
                                    output_size: int = 128) -> np.ndarray:
    """
    Compute multi-scale STFT spectrograms for EGM signals.
    
    Args:
        egm_signals: Array of shape (n_samples, n_channels, n_timesteps).
        frame_lengths: List of STFT frame lengths for each scale.
        frame_steps: List of STFT frame steps for each scale.
        output_size: Resize each spectrogram to (output_size, output_size).
    
    Returns:
        Array of shape (n_samples, output_size, output_size, n_channels*n_scales)
        with concatenated multi-scale spectrograms.
    """
    if frame_lengths is None:
        frame_lengths = [64, 256, 512]
    if frame_steps is None:
        frame_steps = [32, 128, 256]
    
    n_samples, n_channels, n_timesteps = egm_signals.shape
    n_scales = len(frame_lengths)
    
    spectrograms = np.zeros((n_samples, output_size, output_size, n_channels * n_scales))
    
    for sample_idx in range(n_samples):
        spec_idx = 0
        for channel_idx in range(n_channels):
            signal_1d = egm_signals[sample_idx, channel_idx, :]
            
            for scale_idx in range(n_scales):
                spec = compute_stft_spectrogram(
                    signal_1d,
                    frame_length=frame_lengths[scale_idx],
                    frame_step=frame_steps[scale_idx],
                    output_size=output_size
                )
                spectrograms[sample_idx, :, :, spec_idx] = spec
                spec_idx += 1
    
    return spectrograms


def split_data_stratified(X: np.ndarray, 
                         y: np.ndarray,
                         test_ratio: float = 0.2,
                         val_ratio: float = 0.1,
                         random_state: int = 42) -> Tuple[Tuple[np.ndarray, np.ndarray],
                                                           Tuple[np.ndarray, np.ndarray],
                                                           Tuple[np.ndarray, np.ndarray]]:
    """
    Stratified data splitting into train, validation, and test sets.
    
    For multi-label data, uses approximate stratification based on primary label.
    
    Args:
        X: Input features of shape (n_samples, ...).
        y: Binary labels of shape (n_samples, n_classes).
        test_ratio: Fraction of data for test set.
        val_ratio: Fraction of remaining data for validation set.
        random_state: Random seed for reproducibility.
    
    Returns:
        Tuple of ((X_train, y_train), (X_val, y_val), (X_test, y_test))
    """
    # For multi-label, use primary label (first positive class)
    primary_labels = np.argmax(y, axis=1)
    
    # First split: test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, 
        test_size=test_ratio,
        stratify=primary_labels,
        random_state=random_state
    )
    
    # Second split: validation from training
    primary_labels_temp = np.argmax(y_temp, axis=1)
    val_size = val_ratio / (1.0 - test_ratio)  # Adjust ratio
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size,
        stratify=primary_labels_temp,
        random_state=random_state
    )
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def compute_class_weights(y: np.ndarray) -> np.ndarray:
    """
    Compute class weights for imbalanced multi-label data.
    
    Args:
        y: Binary labels of shape (n_samples, n_classes).
    
    Returns:
        Array of class weights.
    """
    # For each class, compute proportion of positive samples
    class_counts = np.sum(y, axis=0)
    n_samples = y.shape[0]
    
    # Weight inversely proportional to class frequency
    weights = n_samples / (2.0 * class_counts + 1e-10)
    
    return weights / np.sum(weights)  # Normalize


def generate_synthetic_egm_data(n_samples: int = 100,
                               n_channels: int = 3,
                               n_timesteps: int = 2500,
                               random_state: int = None) -> np.ndarray:
    """
    Generate synthetic EGM signals with realistic characteristics.
    
    Args:
        n_samples: Number of signals to generate.
        n_channels: Number of channels (unipolar, bipolar, reference).
        n_timesteps: Number of timesteps per signal.
        random_state: Random seed.
    
    Returns:
        Array of shape (n_samples, n_channels, n_timesteps).
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    egm_signals = np.zeros((n_samples, n_channels, n_timesteps))
    
    # Create time vector
    t = np.linspace(0, 2.5, n_timesteps)  # 2.5 seconds at 1kHz
    
    for sample in range(n_samples):
        for channel in range(n_channels):
            # Add multiple frequency components
            freq1 = np.random.uniform(5, 20)  # Cardiac activation
            freq2 = np.random.uniform(1, 5)   # Low frequency baseline
            
            signal_base = 0.5 * np.sin(2 * np.pi * freq1 * t)
            signal_base += 0.2 * np.sin(2 * np.pi * freq2 * t)
            
            # Add noise
            noise_level = np.random.uniform(0.05, 0.15)
            noise = noise_level * np.random.randn(n_timesteps)
            
            # Add occasional spikes (QRS-like features)
            n_spikes = np.random.randint(3, 6)
            for _ in range(n_spikes):
                spike_pos = np.random.randint(0, n_timesteps)
                spike_width = np.random.randint(10, 50)
                egm_signals[sample, channel, spike_pos:spike_pos+spike_width] += \
                    np.random.uniform(0.5, 2.0) * np.exp(-np.linspace(0, 3, spike_width)**2)
            
            # Add high frequency noise (measurement noise)
            hf_noise = 0.1 * np.random.randn(n_timesteps)
            egm_signals[sample, channel, :] = signal_base + noise + hf_noise
    
    return egm_signals


def generate_synthetic_labels(n_samples: int = 100,
                             n_classes: int = 3,
                             class_balance: float = 0.2,
                             random_state: int = None) -> np.ndarray:
    """
    Generate synthetic binary labels for scar tissue localization.
    
    Args:
        n_samples: Number of samples.
        n_classes: Number of label classes.
        class_balance: Fraction of samples with positive labels.
        random_state: Random seed.
    
    Returns:
        Array of shape (n_samples, n_classes) with binary labels.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    labels = np.zeros((n_samples, n_classes), dtype=int)
    
    for class_idx in range(n_classes):
        # Randomly assign positive labels
        n_positive = int(n_samples * class_balance)
        positive_indices = np.random.choice(n_samples, n_positive, replace=False)
        labels[positive_indices, class_idx] = 1
    
    return labels
