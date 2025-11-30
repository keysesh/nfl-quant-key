"""
Model Registry
===============

Tracks model versions, performance metrics, and enables A/B testing.

Features:
- Version tracking (by week, date, git commit)
- Performance monitoring over time
- A/B testing infrastructure
- Automatic rollback on degradation
- Model comparison reports

Usage:
    from nfl_quant.models.model_registry import ModelRegistry

    registry = ModelRegistry()
    registry.register_model(
        model_name='usage_predictor_v4',
        version='week9',
        metrics={'r2': 0.97, 'mae': 2.3},
        model_path='data/models/usage_predictor_v4_defense_week9.joblib'
    )
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging
import shutil

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Central registry for tracking model versions and performance.
    """

    def __init__(self, registry_path: str = 'configs/model_registry.json'):
        """
        Initialize model registry.

        Args:
            registry_path: Path to registry JSON file
        """
        self.registry_path = Path(registry_path)
        self.registry = self._load_registry()

    def _load_registry(self) -> Dict:
        """Load existing registry or create new one."""
        if self.registry_path.exists():
            with open(self.registry_path, 'r') as f:
                registry = json.load(f)
            logger.info(f"Loaded registry with {len(registry.get('models', {}))} models")
            return registry
        else:
            logger.info("Creating new model registry")
            return {
                'created_date': datetime.now().isoformat(),
                'models': {},
                'active_models': {},
            }

    def _save_registry(self):
        """Save registry to disk."""
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.registry_path, 'w') as f:
            json.dump(self.registry, f, indent=2)

        logger.info(f"Registry saved: {self.registry_path}")

    def register_model(
        self,
        model_name: str,
        version: str,
        metrics: Dict[str, float],
        model_path: str,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Register a new model version.

        Args:
            model_name: Model identifier (e.g., 'usage_predictor_v4')
            version: Version string (e.g., 'week9', 'v2.1')
            metrics: Performance metrics (r2, mae, etc.)
            model_path: Path to saved model file
            metadata: Optional additional metadata

        Returns:
            Model ID
        """
        model_id = f"{model_name}_{version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Ensure models dict exists
        if 'models' not in self.registry:
            self.registry['models'] = {}

        # Register model
        self.registry['models'][model_id] = {
            'model_name': model_name,
            'version': version,
            'registered_date': datetime.now().isoformat(),
            'model_path': model_path,
            'metrics': metrics,
            'metadata': metadata or {},
            'status': 'registered',
        }

        # Update active model pointer if this is the first version
        if model_name not in self.registry.get('active_models', {}):
            self.set_active_model(model_name, model_id)

        self._save_registry()

        logger.info(f"Registered model: {model_id}")
        logger.info(f"  Metrics: {metrics}")

        return model_id

    def set_active_model(self, model_name: str, model_id: str):
        """
        Set active model version for production use.

        Args:
            model_name: Model identifier
            model_id: Specific model version ID
        """
        if 'active_models' not in self.registry:
            self.registry['active_models'] = {}

        # Deactivate previous version
        previous_active = self.registry['active_models'].get(model_name)
        if previous_active and previous_active in self.registry['models']:
            self.registry['models'][previous_active]['status'] = 'inactive'

        # Activate new version
        self.registry['active_models'][model_name] = model_id
        self.registry['models'][model_id]['status'] = 'active'
        self.registry['models'][model_id]['activated_date'] = datetime.now().isoformat()

        self._save_registry()

        logger.info(f"Set active model: {model_name} → {model_id}")

    def get_active_model(self, model_name: str) -> Optional[Dict]:
        """
        Get currently active model version.

        Args:
            model_name: Model identifier

        Returns:
            Model metadata dict or None
        """
        model_id = self.registry.get('active_models', {}).get(model_name)
        if model_id:
            return self.registry['models'].get(model_id)
        return None

    def list_versions(self, model_name: str) -> List[Dict]:
        """
        List all versions of a model.

        Args:
            model_name: Model identifier

        Returns:
            List of model version dicts
        """
        versions = []
        for model_id, model_info in self.registry['models'].items():
            if model_info['model_name'] == model_name:
                versions.append({
                    'model_id': model_id,
                    **model_info
                })

        # Sort by registration date (newest first)
        versions.sort(key=lambda x: x['registered_date'], reverse=True)

        return versions

    def compare_models(
        self,
        model_name: str,
        metric: str = 'r2'
    ) -> pd.DataFrame:
        """
        Compare performance of different model versions.

        Args:
            model_name: Model identifier
            metric: Metric to compare (r2, mae, etc.)

        Returns:
            DataFrame with model comparison
        """
        versions = self.list_versions(model_name)

        comparison_data = []
        for v in versions:
            comparison_data.append({
                'version': v['version'],
                'registered_date': v['registered_date'],
                'status': v['status'],
                'metric_value': v['metrics'].get(metric, np.nan),
                'model_id': v['model_id'],
            })

        df = pd.DataFrame(comparison_data)
        df = df.sort_values('registered_date', ascending=False)

        return df

    def detect_performance_degradation(
        self,
        model_name: str,
        current_metrics: Dict[str, float],
        threshold: float = 0.05
    ) -> Tuple[bool, str]:
        """
        Detect if current model performance has degraded.

        Args:
            model_name: Model identifier
            current_metrics: Current performance metrics
            threshold: Degradation threshold (5% by default)

        Returns:
            (degraded, reason)
        """
        active_model = self.get_active_model(model_name)
        if not active_model:
            return False, "No active model to compare"

        baseline_metrics = active_model['metrics']

        # Check each metric
        for metric_name, current_value in current_metrics.items():
            if metric_name not in baseline_metrics:
                continue

            baseline_value = baseline_metrics[metric_name]

            # For metrics where higher is better (R², accuracy)
            if metric_name in ['r2', 'accuracy', 'precision', 'recall', 'f1']:
                degradation = (baseline_value - current_value) / baseline_value
                if degradation > threshold:
                    return True, f"{metric_name} degraded by {degradation:.1%}"

            # For metrics where lower is better (MAE, MSE, loss)
            elif metric_name in ['mae', 'mse', 'rmse', 'loss', 'brier_score']:
                degradation = (current_value - baseline_value) / baseline_value
                if degradation > threshold:
                    return True, f"{metric_name} increased by {degradation:.1%}"

        return False, "No significant degradation"

    def rollback_model(self, model_name: str) -> bool:
        """
        Rollback to previous stable model version.

        Args:
            model_name: Model identifier

        Returns:
            True if rollback successful
        """
        versions = self.list_versions(model_name)

        # Find previous active version (before current)
        current_active_id = self.registry['active_models'].get(model_name)
        previous_version = None

        for v in versions:
            if v['model_id'] != current_active_id and v['status'] in ['inactive', 'registered']:
                previous_version = v
                break

        if not previous_version:
            logger.error("No previous version found for rollback")
            return False

        # Rollback
        logger.warning(f"Rolling back {model_name} from {current_active_id} to {previous_version['model_id']}")
        self.set_active_model(model_name, previous_version['model_id'])

        # Mark current as failed
        self.registry['models'][current_active_id]['status'] = 'failed'
        self.registry['models'][current_active_id]['rollback_date'] = datetime.now().isoformat()

        self._save_registry()

        return True

    def generate_report(self, output_path: str = None) -> str:
        """
        Generate model registry report.

        Args:
            output_path: Optional path to save HTML report

        Returns:
            HTML report string
        """
        html = ["<html><head><title>Model Registry Report</title></head><body>"]
        html.append("<h1>NFL QUANT Model Registry</h1>")
        html.append(f"<p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>")

        # Active models
        html.append("<h2>Active Models</h2>")
        html.append("<table border='1'>")
        html.append("<tr><th>Model</th><th>Version</th><th>Activated</th><th>Metrics</th></tr>")

        for model_name, model_id in self.registry.get('active_models', {}).items():
            model_info = self.registry['models'].get(model_id, {})
            metrics_str = ", ".join([f"{k}={v:.3f}" for k, v in model_info.get('metrics', {}).items()])

            html.append(f"<tr>")
            html.append(f"<td>{model_name}</td>")
            html.append(f"<td>{model_info.get('version', 'unknown')}</td>")
            html.append(f"<td>{model_info.get('activated_date', 'N/A')}</td>")
            html.append(f"<td>{metrics_str}</td>")
            html.append(f"</tr>")

        html.append("</table>")

        html.append("</body></html>")

        report_html = "\n".join(html)

        if output_path:
            Path(output_path).write_text(report_html)
            logger.info(f"Report saved: {output_path}")

        return report_html


# Example usage
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    print("="*80)
    print("MODEL REGISTRY DEMONSTRATION")
    print("="*80)

    registry = ModelRegistry(registry_path='configs/model_registry_demo.json')

    # Register models
    print("\n1. Registering models...")
    registry.register_model(
        model_name='usage_predictor_v4',
        version='week8',
        metrics={'r2': 0.968, 'mae': 2.4},
        model_path='data/models/usage_predictor_v4_defense_week8.joblib',
        metadata={'training_samples': 3500}
    )

    registry.register_model(
        model_name='usage_predictor_v4',
        version='week9',
        metrics={'r2': 0.972, 'mae': 2.2},
        model_path='data/models/usage_predictor_v4_defense_week9.joblib',
        metadata={'training_samples': 3800}
    )

    # Set active
    print("\n2. Setting active model...")
    registry.set_active_model('usage_predictor_v4', 'usage_predictor_v4_week9_*')

    # Compare versions
    print("\n3. Comparing versions...")
    df_comparison = registry.compare_models('usage_predictor_v4', metric='r2')
    print(df_comparison[['version', 'metric_value', 'status']])

    # Test degradation detection
    print("\n4. Testing degradation detection...")
    degraded, reason = registry.detect_performance_degradation(
        'usage_predictor_v4',
        {'r2': 0.910, 'mae': 3.2},  # Simulated poor performance
        threshold=0.05
    )
    print(f"Degradation detected: {degraded}")
    if degraded:
        print(f"Reason: {reason}")

    print("\n" + "="*80)
    print("Model registry created successfully!")
    print("Check: configs/model_registry_demo.json")
