"""
Utility functions for visualization, evaluation, and common operations.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import json
import os


def log_metrics(metrics: Dict[str, float], output_file: Optional[str] = None) -> None:
    """
    Log evaluation metrics.
    
    Args:
        metrics: Dictionary of metric names and values.
        output_file: Optional file to write metrics to.
    """
    print("\n" + "="*50)
    print("EVALUATION METRICS")
    print("="*50)
    for name, value in metrics.items():
        if isinstance(value, float):
            print(f"{name:.<40} {value:>8.4f}")
        else:
            print(f"{name:.<40} {value:>8}")
    print("="*50 + "\n")
    
    if output_file:
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {output_file}")


def compute_sample_statistics(data: np.ndarray) -> Dict[str, float]:
    """
    Compute basic statistics of a data array.
    
    Args:
        data: Input array.
    
    Returns:
        Dictionary with statistics.
    """
    return {
        'mean': float(np.mean(data)),
        'std': float(np.std(data)),
        'min': float(np.min(data)),
        'max': float(np.max(data)),
        'median': float(np.median(data)),
        'q1': float(np.percentile(data, 25)),
        'q3': float(np.percentile(data, 75)),
    }


def print_batch_info(X: np.ndarray, y: np.ndarray, batch_name: str = "Batch") -> None:
    """
    Print information about a batch.
    
    Args:
        X: Input features.
        y: Labels.
        batch_name: Name of batch for printing.
    """
    print(f"\n{batch_name} Information:")
    print(f"  Input shape: {X.shape}")
    print(f"  Label shape: {y.shape}")
    print(f"  Input statistics: {compute_sample_statistics(X)}")
    if len(y.shape) > 1:
        print(f"  Label distribution: {np.sum(y, axis=0)}")


def ensure_output_directory(output_dir: str) -> str:
    """
    Ensure output directory exists.
    
    Args:
        output_dir: Path to directory.
    
    Returns:
        The output directory path.
    """
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def save_model_config(model_config: Dict, output_path: str) -> None:
    """
    Save model configuration to JSON file.
    
    Args:
        model_config: Model configuration dictionary.
        output_path: Path where to save.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    # Convert non-serializable types
    config_copy = json.loads(json.dumps(model_config, default=str))
    with open(output_path, "w") as f:
        json.dump(config_copy, f, indent=2)
    print(f"Model config saved to {output_path}")


class DataNormalizer:
    """Helper class for consistent data normalization."""
    
    def __init__(self, method: str = "peak"):
        """
        Initialize normalizer.
        
        Args:
            method: "peak" for peak normalization, "std" for z-score.
        """
        self.method = method
        self.fitted_params = {}
    
    def fit(self, X: np.ndarray) -> None:
        """
        Estimate normalization parameters from data.
        
        Args:
            X: Training data of shape (n_samples, n_channels, n_timesteps).
        """
        if self.method == "peak":
            self.fitted_params['max_vals'] = np.max(np.abs(X), axis=(0, 2))
        elif self.method == "std":
            self.fitted_params['mean_vals'] = np.mean(X, axis=(0, 2))
            self.fitted_params['std_vals'] = np.std(X, axis=(0, 2))
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply normalization to data.
        
        Args:
            X: Data to normalize.
        
        Returns:
            Normalized data.
        """
        X_norm = X.copy()
        
        if self.method == "peak":
            max_vals = self.fitted_params.get('max_vals', np.ones(X.shape[1]))
            for i in range(X.shape[1]):
                if max_vals[i] > 0:
                    X_norm[:, i, :] /= max_vals[i]
        elif self.method == "std":
            mean_vals = self.fitted_params.get('mean_vals', np.zeros(X.shape[1]))
            std_vals = self.fitted_params.get('std_vals', np.ones(X.shape[1]))
            for i in range(X.shape[1]):
                X_norm[:, i, :] = (X_norm[:, i, :] - mean_vals[i]) / (std_vals[i] + 1e-10)
        
        return X_norm
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(X)
        return self.transform(X)
