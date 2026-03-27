"""
Configuration module for EGM to Scar Mapping project.

This module handles loading and managing configuration settings for the project,
including data paths, model parameters, and processing options.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class Config:
    """Configuration manager for EGM to Scar Mapping project."""
    
    # Default data paths (relative to project root)
    DEFAULT_DATA_DIR = "data/sample_data"
    DEFAULT_OUTPUT_DIR = "output"
    DEFAULT_MODEL_DIR = "models"
    
    # Model parameters
    DEFAULT_MODEL_PARAMS = {
        "cnn_stft": {
            "batch_size": 32,
            "epochs": 10,
            "learning_rate": 1e-5,
            "early_stopping_patience": 10,
            "stft_scales": 3,
            "stft_frame_sizes": [64, 256, 512],
            "stft_hop_sizes": [32, 128, 256],
            "spectrogram_size": 128,
            "initial_filters": 16,
            "dropout_rate": 0.2,
            "l2_regularization": 1e-4,
        },
        "transformer": {
            "batch_size": 32,
            "epochs": 50,
            "max_learning_rate": 5e-5,
            "warmup_steps": 1000,
            "patch_size": 50,
            "d_model": 256,
            "nhead": 8,
            "num_encoder_layers": 8,
            "dim_feedforward": 1024,
            "dropout": 0.1,
            "gradient_clip_norm": 1.0,
            "loss_weights": {
                "cross_channel": 0.3,
                "intra_channel": 0.3,
                "autoregressive": 0.3,
                "contrastive": 0.1
            },
            "regularization_weights": {
                "smoothness": 10.0,
                "curvature": 10.0,
                "mae": 1.0,
                "stft": 0.1
            }
        }
    }
    
    # Data processing parameters
    DEFAULT_DATA_PARAMS = {
        "egm_sampling_rate": 1000,  # Hz
        "egm_duration": 2.5,  # seconds
        "egm_n_samples": 2500,  # 2.5s @ 1000Hz
        "egm_n_channels": 3,  # unipolar, bipolar, reference
        "test_set_ratio": 0.2,
        "val_set_ratio": 0.1,
        "n_label_classes": 3,  # endocardial, mid-myocardial, epicardial
        "label_names": ["endocardial", "mid_myocardial", "epicardial"],
    }
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration.
        
        Args:
            config_path: Optional path to a YAML configuration file.
                        If None, uses default values.
        """
        self.data_params = self.DEFAULT_DATA_PARAMS.copy()
        self.model_params = self.DEFAULT_MODEL_PARAMS.copy()
        self.paths = self._initialize_paths()
        
        if config_path and os.path.exists(config_path):
            self.load_from_yaml(config_path)
    
    def _initialize_paths(self) -> Dict[str, Path]:
        """Initialize project directory paths."""
        project_root = self._get_project_root()
        return {
            "project_root": project_root,
            "data_dir": project_root / self.DEFAULT_DATA_DIR,
            "output_dir": project_root / self.DEFAULT_OUTPUT_DIR,
            "model_dir": project_root / self.DEFAULT_MODEL_DIR,
            "notebook_dir": project_root / "notebooks",
        }
    
    @staticmethod
    def _get_project_root() -> Path:
        """Get the project root directory."""
        current_file = Path(__file__).resolve()
        return current_file.parent.parent
    
    def load_from_yaml(self, config_path: str) -> None:
        """
        Load configuration from a YAML file.
        
        Args:
            config_path: Path to YAML configuration file.
        """
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f) or {}
        
        if 'data_params' in config_data:
            self.data_params.update(config_data['data_params'])
        if 'model_params' in config_data:
            self.model_params.update(config_data['model_params'])
        if 'paths' in config_data:
            custom_paths = config_data['paths']
            for key, value in custom_paths.items():
                self.paths[key] = Path(value)
    
    def load_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """
        Load configuration from a dictionary.
        
        Args:
            config_dict: Dictionary with configuration settings.
        """
        if 'data_params' in config_dict:
            self.data_params.update(config_dict['data_params'])
        if 'model_params' in config_dict:
            self.model_params.update(config_dict['model_params'])
        if 'paths' in config_dict:
            custom_paths = config_dict['paths']
            for key, value in custom_paths.items():
                self.paths[key] = Path(value)
    
    def get_data_param(self, key: str, default: Any = None) -> Any:
        """Get a data parameter by key."""
        return self.data_params.get(key, default)
    
    def get_model_param(self, model_name: str, key: str, default: Any = None) -> Any:
        """Get a model parameter."""
        if model_name in self.model_params:
            return self.model_params[model_name].get(key, default)
        return default
    
    def ensure_directories_exist(self) -> None:
        """Create all configured directories if they don't exist."""
        for path in self.paths.values():
            if isinstance(path, Path) and path.name != 'project_root':
                path.mkdir(parents=True, exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "data_params": self.data_params,
            "model_params": self.model_params,
            "paths": {k: str(v) for k, v in self.paths.items()}
        }
    
    def __repr__(self) -> str:
        return f"Config(data_params={len(self.data_params)}, model_params={len(self.model_params)})"


# Global configuration instance
_config_instance = None


def get_config(config_path: Optional[str] = None) -> Config:
    """
    Get or create the global configuration instance.
    
    Args:
        config_path: Optional path to configuration YAML file.
    
    Returns:
        Config instance.
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = Config(config_path)
    return _config_instance


def reset_config() -> None:
    """Reset the global configuration instance."""
    global _config_instance
    _config_instance = None
