"""
Data loading utilities for model development, training, and evaluation.

This module provides functions to load preprocessed EGM datasets and small
supporting artifacts used by the published modeling workflow.
"""

import os
import pickle
import json
import numpy as np
from typing import Dict, Tuple, Any


def load_egm_data(data_dir: str) -> Tuple[Dict, Dict]:
    """
    Load EGM signals and labels from preprocessed .npy files.
    
    Args:
        data_dir: Directory containing egm_signals.npy and egm_labels.npy files.
    
    Returns:
        Tuple of (signals_dict, labels_dict) containing loaded arrays.
    
    Raises:
        FileNotFoundError: If required data files are not found.
    """
    signals_path = os.path.join(data_dir, "egm_signals.npy")
    labels_path = os.path.join(data_dir, "egm_labels.npy")
    
    if not os.path.exists(signals_path):
        raise FileNotFoundError(f"Signals file not found: {signals_path}")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    
    signals = np.load(signals_path)
    labels = np.load(labels_path)
    
    return {
        "signals": signals,
        "path": signals_path
    }, {
        "labels": labels,
        "path": labels_path
    }


def load_processed_datasets(data_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
                                                     np.ndarray, np.ndarray, np.ndarray]:
    """
    Load preprocessed train/val/test datasets for model training.
    
    Args:
        data_dir: Directory containing X_train.npy, y_train.npy, etc.
    
    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    X_val = np.load(os.path.join(data_dir, "X_val.npy"))
    y_val = np.load(os.path.join(data_dir, "y_val.npy"))
    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def load_filtered_datasets(data_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load filtered preprocessed datasets (e.g., after bandpass filtering).
    
    Args:
        data_dir: Directory containing filtered .npy files.
    
    Returns:
        Tuple of (X_train_filtered, X_val_filtered, X_test_filtered)
    """
    X_train_filtered = np.load(os.path.join(data_dir, "X_train_filtered.npy"))
    X_val_filtered = np.load(os.path.join(data_dir, "X_val_filtered.npy"))
    X_test_filtered = np.load(os.path.join(data_dir, "X_test_filtered.npy"))
    
    return X_train_filtered, X_val_filtered, X_test_filtered


def load_pickle_file(file_path: str) -> Any:
    """
    Load data from a pickle file.
    
    Args:
        file_path: Path to pickle file.
    
    Returns:
        Unpickled Python object.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Pickle file not found: {file_path}")
    
    with open(file_path, "rb") as f:
        return pickle.load(f)


def save_pickle_file(data: Any, file_path: str) -> None:
    """
    Save data to a pickle file.
    
    Args:
        data: Python object to pickle.
        file_path: Path where to save the pickle file.
    """
    os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
    with open(file_path, "wb") as f:
        pickle.dump(data, f)


def load_json_file(file_path: str) -> Dict:
    """
    Load JSON configuration file.
    
    Args:
        file_path: Path to JSON file.
    
    Returns:
        Dictionary from JSON file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"JSON file not found: {file_path}")
    
    with open(file_path, "r") as f:
        return json.load(f)


def save_json_file(data: Dict, file_path: str, indent: int = 2) -> None:
    """
    Save data to a JSON file.
    
    Args:
        data: Dictionary to save.
        file_path: Path where to save the JSON file.
        indent: JSON indentation level.
    """
    os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(data, f, indent=indent)


def load_offset_matrix(file_path: str) -> np.ndarray:
    """
    Load coordinate transformation matrix from JSON.
    
    Args:
        file_path: Path to JSON file containing affine matrix.
    
    Returns:
        4x4 affine transformation matrix as numpy array.
    """
    offset_data = load_json_file(file_path)
    return np.array(offset_data.get("offset_matrix", np.eye(4)))
