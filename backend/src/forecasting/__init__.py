"""
Forecasting module for supply chain predictions.
"""
from .model import BaselineForecaster
from .model_manager import ModelManager, get_model_manager

__all__ = ["BaselineForecaster", "ModelManager", "get_model_manager"]

