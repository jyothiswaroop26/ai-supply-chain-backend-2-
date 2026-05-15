"""
Model Manager for Supply Chain Forecasting Models.

This module provides utilities for managing model lifecycle including:
- Saving trained models with metadata
- Loading saved models
- Model versioning and registry
- Model metadata tracking
"""

import logging
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any, Union

import joblib

from .model import BaselineForecaster
from .models.random_forest import RandomForestForecaster

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Central manager for model persistence and lifecycle management.
    
    Handles saving/loading models with metadata tracking and versioning.
    """
    
    def __init__(self, models_dir: Optional[str] = None):
        """
        Initialize ModelManager.
        
        Args:
            models_dir: Directory to store models. Defaults to ./models/
        """
        if models_dir is None:
            models_dir = str(Path(__file__).parent.parent.parent / "models")
        
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.registry_file = self.models_dir / "model_registry.json"
        self.registry = self._load_registry()
        
        logger.info(f"ModelManager initialized. Models directory: {self.models_dir}")
    
    def _load_registry(self) -> Dict[str, Dict]:
        """Load model registry from file."""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load registry: {e}")
                return {}
        return {}
    
    def _save_registry(self) -> None:
        """Save model registry to file."""
        try:
            with open(self.registry_file, 'w') as f:
                json.dump(self.registry, f, indent=2, default=str)
            logger.info("Model registry saved")
        except Exception as e:
            logger.error(f"Failed to save registry: {e}")
    
    def save_baseline_model(self,
                           model: BaselineForecaster,
                           model_name: str,
                           metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Save a BaselineForecaster model.
        
        Args:
            model: Trained BaselineForecaster instance
            model_name: Identifier for the model
            metadata: Optional metadata dictionary
        
        Returns:
            Path to saved model
        """
        model_dir = self.models_dir / "baseline" / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = model_dir / f"{model_name}_model.joblib"
        meta_path = model_dir / "metadata.json"
        
        # Save model
        model.save_model(str(model_path))
        
        # Save metadata
        meta = {
            'model_type': 'BaselineForecaster',
            'model_name': model_name,
            'saved_at': datetime.now().isoformat(),
            'model_info': model.get_model_info(),
            'custom_metadata': metadata or {}
        }
        
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2, default=str)
        
        # Update registry
        registry_key = f"baseline/{model_name}"
        self.registry[registry_key] = {
            'type': 'baseline',
            'path': str(model_path),
            'metadata_path': str(meta_path),
            'saved_at': datetime.now().isoformat()
        }
        self._save_registry()
        
        logger.info(f"BaselineForecaster model saved: {model_path}")
        return str(model_path)
    
    def save_random_forest_model(self,
                                model: RandomForestForecaster,
                                model_name: str,
                                metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Save a RandomForestForecaster model.
        
        Args:
            model: Trained RandomForestForecaster instance
            model_name: Identifier for the model
            metadata: Optional metadata dictionary
        
        Returns:
            Path to saved model
        """
        model_dir = self.models_dir / "random_forest" / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = model_dir / f"{model_name}_model.joblib"
        meta_path = model_dir / "metadata.json"
        
        # Save model
        model.save_model(str(model_path))
        
        # Save metadata
        meta = {
            'model_type': 'RandomForestForecaster',
            'model_name': model_name,
            'saved_at': datetime.now().isoformat(),
            'model_info': model.get_model_info(),
            'training_metrics': model.get_training_history(),
            'custom_metadata': metadata or {}
        }
        
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2, default=str)
        
        # Update registry
        registry_key = f"random_forest/{model_name}"
        self.registry[registry_key] = {
            'type': 'random_forest',
            'path': str(model_path),
            'metadata_path': str(meta_path),
            'saved_at': datetime.now().isoformat()
        }
        self._save_registry()
        
        logger.info(f"RandomForestForecaster model saved: {model_path}")
        return str(model_path)
    
    def load_baseline_model(self, model_name: str) -> BaselineForecaster:
        """
        Load a saved BaselineForecaster model.
        
        Args:
            model_name: Identifier of the model
        
        Returns:
            Loaded BaselineForecaster instance
        """
        registry_key = f"baseline/{model_name}"
        
        if registry_key not in self.registry:
            raise ValueError(f"Model '{model_name}' not found in registry")
        
        model_path = self.registry[registry_key]['path']
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model = BaselineForecaster()
        model.load_model(model_path)
        
        logger.info(f"BaselineForecaster model loaded: {model_path}")
        return model
    
    def load_random_forest_model(self, model_name: str) -> RandomForestForecaster:
        """
        Load a saved RandomForestForecaster model.
        
        Args:
            model_name: Identifier of the model
        
        Returns:
            Loaded RandomForestForecaster instance
        """
        registry_key = f"random_forest/{model_name}"
        
        if registry_key not in self.registry:
            raise ValueError(f"Model '{model_name}' not found in registry")
        
        model_path = self.registry[registry_key]['path']
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model = RandomForestForecaster()
        model.load_model(model_path)
        
        logger.info(f"RandomForestForecaster model loaded: {model_path}")
        return model
    
    def get_model_metadata(self, model_name: str, model_type: str = 'baseline') -> Dict:
        """
        Get metadata for a saved model.
        
        Args:
            model_name: Identifier of the model
            model_type: Type of model ('baseline' or 'random_forest')
        
        Returns:
            Dictionary with model metadata
        """
        registry_key = f"{model_type}/{model_name}"
        
        if registry_key not in self.registry:
            raise ValueError(f"Model '{model_name}' not found in registry")
        
        meta_path = self.registry[registry_key]['metadata_path']
        
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")
        
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
        
        return metadata
    
    def list_models(self) -> Dict[str, list]:
        """
        List all saved models organized by type.
        
        Returns:
            Dictionary with models grouped by type
        """
        models_by_type = {'baseline': [], 'random_forest': []}
        
        for registry_key in self.registry.keys():
            model_type, model_name = registry_key.split('/', 1)
            models_by_type[model_type].append(model_name)
        
        logger.info(f"Found {len(self.registry)} models in registry")
        return models_by_type
    
    def delete_model(self, model_name: str, model_type: str = 'baseline') -> None:
        """
        Delete a saved model.
        
        Args:
            model_name: Identifier of the model
            model_type: Type of model ('baseline' or 'random_forest')
        """
        registry_key = f"{model_type}/{model_name}"
        
        if registry_key not in self.registry:
            raise ValueError(f"Model '{model_name}' not found in registry")
        
        model_path = Path(self.registry[registry_key]['path'])
        meta_path = Path(self.registry[registry_key]['metadata_path'])
        
        # Delete files
        if model_path.exists():
            model_path.unlink()
        if meta_path.exists():
            meta_path.unlink()
        
        # Clean up empty directories
        try:
            model_path.parent.rmdir()
            model_path.parent.parent.rmdir()
        except OSError:
            pass  # Directory not empty or other error
        
        # Update registry
        del self.registry[registry_key]
        self._save_registry()
        
        logger.info(f"Model deleted: {model_name}")
    
    def export_model_info(self) -> Dict:
        """
        Export complete model registry information.
        
        Returns:
            Dictionary with all model information
        """
        export_data = {
            'exported_at': datetime.now().isoformat(),
            'models_directory': str(self.models_dir),
            'models': {}
        }
        
        for registry_key in self.registry.keys():
            model_type, model_name = registry_key.split('/', 1)
            try:
                metadata = self.get_model_metadata(model_name, model_type)
                export_data['models'][registry_key] = metadata
            except Exception as e:
                logger.error(f"Failed to export info for {registry_key}: {e}")
        
        return export_data


# Convenience functions
def get_model_manager(models_dir: Optional[str] = None) -> ModelManager:
    """Get or create a ModelManager instance."""
    return ModelManager(models_dir)
