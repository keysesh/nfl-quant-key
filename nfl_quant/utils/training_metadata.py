"""
Training Metadata Utility

Ensures all trained models include metadata about:
- Training data cutoff (max_week, max_season)
- Training timestamp
- Git commit hash (if available)
- Feature list
- Training parameters

This prevents data leakage by making training data boundaries explicit.
"""

import joblib
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
import json


def get_git_hash() -> str:
    """Get current git commit hash, or 'unknown' if not in a git repo."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True, text=True, cwd=Path(__file__).parent.parent.parent
        )
        return result.stdout.strip()[:8] if result.returncode == 0 else 'unknown'
    except Exception:
        return 'unknown'


def save_model_with_metadata(
    model: Any,
    path: Path,
    training_data_cutoff: Dict[str, int],
    feature_names: Optional[list] = None,
    training_params: Optional[Dict] = None,
    notes: str = ""
) -> Path:
    """
    Save a model with embedded training metadata.

    Args:
        model: The trained model object
        path: Where to save the .joblib file
        training_data_cutoff: Dict with 'max_season' and 'max_week' keys
            Example: {'max_season': 2025, 'max_week': 11}
        feature_names: List of feature names used in training
        training_params: Dict of hyperparameters used
        notes: Any additional notes about training

    Returns:
        Path to saved model

    Example:
        >>> save_model_with_metadata(
        ...     model=xgb_classifier,
        ...     path=Path('data/models/my_model.joblib'),
        ...     training_data_cutoff={'max_season': 2025, 'max_week': 11},
        ...     feature_names=['feature1', 'feature2'],
        ...     notes="Trained for week 12+ predictions"
        ... )
    """
    if 'max_season' not in training_data_cutoff or 'max_week' not in training_data_cutoff:
        raise ValueError("training_data_cutoff must include 'max_season' and 'max_week'")

    metadata = {
        'training_timestamp': datetime.now().isoformat(),
        'git_hash': get_git_hash(),
        'training_data_cutoff': training_data_cutoff,
        'feature_names': feature_names or [],
        'training_params': training_params or {},
        'notes': notes,
        'model_type': type(model).__name__
    }

    # Wrap model with metadata
    wrapped = {
        'model': model,
        '_training_metadata': metadata
    }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(wrapped, path)

    # Also save metadata as sidecar JSON for easy inspection
    metadata_path = path.with_suffix('.metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved model to {path}")
    print(f"  Training data cutoff: season {training_data_cutoff['max_season']}, week {training_data_cutoff['max_week']}")
    print(f"  Metadata saved to {metadata_path}")

    return path


def load_model_with_metadata(path: Path) -> tuple:
    """
    Load a model and its training metadata.

    Args:
        path: Path to .joblib file

    Returns:
        Tuple of (model, metadata_dict)
        If model was saved without metadata, metadata will be empty dict.

    Example:
        >>> model, metadata = load_model_with_metadata(Path('data/models/my_model.joblib'))
        >>> print(metadata['training_data_cutoff'])
        {'max_season': 2025, 'max_week': 11}
    """
    path = Path(path)
    loaded = joblib.load(path)

    # Check if wrapped with metadata
    if isinstance(loaded, dict) and '_training_metadata' in loaded:
        return loaded['model'], loaded['_training_metadata']

    # Legacy model without metadata - try to load sidecar JSON
    metadata_path = path.with_suffix('.metadata.json')
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            return loaded, json.load(f)

    # No metadata available
    return loaded, {}


def verify_training_cutoff(
    path: Path,
    required_max_week: int,
    required_max_season: int = 2025
) -> bool:
    """
    Verify a model was trained on data before the required cutoff.

    Args:
        path: Path to model file
        required_max_week: The maximum week that training data should have included
        required_max_season: The maximum season (default 2025)

    Returns:
        True if model is valid for use, False if it may have data leakage

    Raises:
        ValueError if model has no metadata (cannot verify)
    """
    _, metadata = load_model_with_metadata(path)

    if not metadata:
        raise ValueError(f"Model {path} has no training metadata - cannot verify cutoff")

    cutoff = metadata.get('training_data_cutoff', {})
    model_season = cutoff.get('max_season', 0)
    model_week = cutoff.get('max_week', 0)

    # Valid if model's training data ended before required cutoff
    if model_season < required_max_season:
        return True
    if model_season == required_max_season and model_week <= required_max_week:
        return True

    print(f"WARNING: Model {path.name} may have data leakage!")
    print(f"  Model trained on: season {model_season}, week {model_week}")
    print(f"  Required cutoff: season {required_max_season}, week {required_max_week}")
    return False
