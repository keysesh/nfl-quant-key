"""
Version-Free Classifier Registry
================================

Manages interaction classifier models using feature fingerprints instead of version numbers.
Models are identified by their feature set + markets, not arbitrary version strings.

Key Concepts:
- fingerprint: 8-char MD5 hash of sorted(features) + sorted(markets)
- model_id: "model_YYYYMMDD_HHMM_fingerprint" - unique, human-readable
- registry: JSON file tracking all models and which is active

Usage:
    from nfl_quant.models.classifier_registry import (
        register_model,
        get_active_model,
        get_active_model_info,
        validate_model_compatibility
    )

    # Register after training
    model_id = register_model(model_data, features, markets, metrics, set_active=True)

    # Load for inference
    model_data = get_active_model()
"""

import hashlib
import json
import joblib
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = PROJECT_ROOT / 'data' / 'models'
REGISTRY_PATH = MODELS_DIR / 'model_registry.json'
ACTIVE_MODEL_PATH = MODELS_DIR / 'active_model.joblib'


def get_feature_fingerprint(features: List[str], markets: List[str]) -> str:
    """
    Generate 8-character fingerprint from features and markets.

    This uniquely identifies a model configuration without needing version numbers.
    Same features + markets = same fingerprint, regardless of when trained.

    Args:
        features: List of feature column names
        markets: List of market names

    Returns:
        8-character hex string (first 8 chars of MD5 hash)
    """
    # Sort for consistency
    feature_str = ','.join(sorted(features))
    market_str = ','.join(sorted(markets))
    combined = f"{feature_str}|{market_str}"

    return hashlib.md5(combined.encode()).hexdigest()[:8]


def generate_model_id(features: List[str], markets: List[str]) -> str:
    """
    Generate unique model ID: "model_YYYYMMDD_HHMM_fingerprint"

    Args:
        features: List of feature column names
        markets: List of market names

    Returns:
        Unique model identifier string
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    fingerprint = get_feature_fingerprint(features, markets)
    return f"model_{timestamp}_{fingerprint}"


def _load_registry() -> Dict:
    """Load registry JSON, creating if needed."""
    if REGISTRY_PATH.exists():
        try:
            with open(REGISTRY_PATH, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.warning(f"Invalid registry JSON at {REGISTRY_PATH}, creating new")

    return {
        "active": None,
        "models": {},
        "created": datetime.now().isoformat()
    }


def _save_registry(registry: Dict):
    """Save registry to JSON."""
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_PATH, 'w') as f:
        json.dump(registry, f, indent=2)


def register_model(
    model_data: Dict,
    features: List[str],
    markets: List[str],
    metrics: Optional[Dict] = None,
    set_active: bool = True,
    notes: str = ""
) -> str:
    """
    Register a trained model in the registry.

    Args:
        model_data: Dict containing 'models' (per-market classifiers) and other metadata
        features: List of feature column names used in training
        markets: List of market names the model supports
        metrics: Optional performance metrics (ROI, hit_rate, etc.)
        set_active: If True, set this as the active model
        notes: Optional description/notes about the model

    Returns:
        Generated model_id
    """
    model_id = generate_model_id(features, markets)
    fingerprint = get_feature_fingerprint(features, markets)

    # Save model file with fingerprint in name for tracking
    model_path = MODELS_DIR / f"{model_id}.joblib"

    # Ensure model_data has the right structure
    model_data['feature_cols'] = features
    model_data['feature_count'] = len(features)
    model_data['fingerprint'] = fingerprint
    model_data['model_id'] = model_id
    model_data['registered'] = datetime.now().isoformat()

    # Save model
    joblib.dump(model_data, model_path)
    logger.info(f"Saved model to {model_path}")

    # Update registry
    registry = _load_registry()

    registry['models'][model_id] = {
        'created': datetime.now().isoformat(),
        'features': features,
        'markets': markets,
        'fingerprint': fingerprint,
        'metrics': metrics or {},
        'path': str(model_path.relative_to(PROJECT_ROOT)),
        'notes': notes,
        'feature_count': len(features)
    }

    if set_active:
        registry['active'] = model_id
        # Also copy to active_model.joblib for backward compatibility
        joblib.dump(model_data, ACTIVE_MODEL_PATH)
        logger.info(f"Set {model_id} as active model")

    _save_registry(registry)

    return model_id


def get_active_model() -> Optional[Dict]:
    """
    Get the currently active model.

    Returns:
        Dict with model data including:
        - models: Dict of per-market XGBoost classifiers
        - feature_cols: List of feature names
        - fingerprint: 8-char feature fingerprint
        - model_id: Unique model identifier

    Falls back to active_model.joblib if registry not available.
    """
    # Try registry first
    registry = _load_registry()
    active_id = registry.get('active')

    if active_id and active_id in registry.get('models', {}):
        model_info = registry['models'][active_id]
        model_path = PROJECT_ROOT / model_info['path']

        if model_path.exists():
            model_data = joblib.load(model_path)
            # Ensure it has all expected fields
            model_data['model_id'] = active_id
            model_data['fingerprint'] = model_info['fingerprint']
            return model_data

    # Fallback: load active_model.joblib directly
    if ACTIVE_MODEL_PATH.exists():
        logger.info("Loading from active_model.joblib (legacy fallback)")
        model_data = joblib.load(ACTIVE_MODEL_PATH)

        # Add fingerprint if missing
        if 'fingerprint' not in model_data:
            features = model_data.get('feature_cols', [])
            markets = list(model_data.get('models', {}).keys())
            model_data['fingerprint'] = get_feature_fingerprint(features, markets)
            model_data['model_id'] = f"legacy_{model_data.get('version', 'unknown')}"

        return model_data

    logger.error("No active model found")
    return None


def get_active_model_info() -> Optional[Dict]:
    """
    Get metadata about active model without loading full model.

    Returns:
        Dict with model metadata (features, markets, fingerprint, metrics)
        or None if no active model.
    """
    registry = _load_registry()
    active_id = registry.get('active')

    if active_id and active_id in registry.get('models', {}):
        return {
            'model_id': active_id,
            **registry['models'][active_id]
        }

    # Fallback: get info from active_model.joblib
    if ACTIVE_MODEL_PATH.exists():
        model_data = joblib.load(ACTIVE_MODEL_PATH)
        features = model_data.get('feature_cols', [])
        markets = list(model_data.get('models', {}).keys())

        return {
            'model_id': f"legacy_{model_data.get('version', 'unknown')}",
            'features': features,
            'markets': markets,
            'fingerprint': get_feature_fingerprint(features, markets),
            'metrics': model_data.get('metrics', {}),
            'feature_count': len(features)
        }

    return None


def validate_model_compatibility(features: List[str], markets: List[str]) -> bool:
    """
    Check if given features/markets match the active model.

    Args:
        features: List of feature names to check
        markets: List of market names to check

    Returns:
        True if compatible, False otherwise
    """
    model_info = get_active_model_info()
    if not model_info:
        return False

    # Check fingerprints match
    input_fingerprint = get_feature_fingerprint(features, markets)
    model_fingerprint = model_info.get('fingerprint')

    return input_fingerprint == model_fingerprint


def list_models() -> List[Dict]:
    """
    List all registered models.

    Returns:
        List of model info dicts, sorted by creation date (newest first)
    """
    registry = _load_registry()
    active_id = registry.get('active')

    models = []
    for model_id, info in registry.get('models', {}).items():
        models.append({
            'model_id': model_id,
            'is_active': model_id == active_id,
            **info
        })

    # Sort by created date, newest first
    models.sort(key=lambda x: x.get('created', ''), reverse=True)
    return models


def set_active_model(model_id: str) -> bool:
    """
    Set a registered model as active.

    Args:
        model_id: Model identifier to activate

    Returns:
        True if successful, False otherwise
    """
    registry = _load_registry()

    if model_id not in registry.get('models', {}):
        logger.error(f"Model {model_id} not found in registry")
        return False

    model_info = registry['models'][model_id]
    model_path = PROJECT_ROOT / model_info['path']

    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        return False

    # Load and copy to active_model.joblib
    model_data = joblib.load(model_path)
    joblib.dump(model_data, ACTIVE_MODEL_PATH)

    # Update registry
    registry['active'] = model_id
    _save_registry(registry)

    logger.info(f"Set {model_id} as active model")
    return True


def migrate_legacy_model() -> Optional[str]:
    """
    Migrate existing active_model.joblib to registry format.

    Returns:
        model_id if successful, None otherwise
    """
    if not ACTIVE_MODEL_PATH.exists():
        logger.warning("No active_model.joblib to migrate")
        return None

    model_data = joblib.load(ACTIVE_MODEL_PATH)

    features = model_data.get('feature_cols', [])
    markets = list(model_data.get('models', {}).keys())
    metrics = model_data.get('metrics', {})
    version = model_data.get('version', 'unknown')

    if not features or not markets:
        logger.error("Cannot migrate: missing features or markets")
        return None

    model_id = register_model(
        model_data=model_data,
        features=features,
        markets=markets,
        metrics=metrics,
        set_active=True,
        notes=f"Migrated from {version}"
    )

    logger.info(f"Migrated legacy model to {model_id}")
    return model_id


# Convenience function for backward compatibility
def load_active_model() -> Optional[Dict]:
    """Alias for get_active_model() for backward compatibility."""
    return get_active_model()
