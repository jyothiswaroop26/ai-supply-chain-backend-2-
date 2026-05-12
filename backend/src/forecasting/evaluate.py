"""
Evaluation metrics module for supply chain forecasting models.

This module provides comprehensive evaluation metrics including:
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² Score
- MAPE (Mean Absolute Percentage Error)
"""
import logging
from typing import Dict, Tuple, Union, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error
)

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Comprehensive evaluation metrics for forecasting models.
    
    Supports single and multiple step ahead forecasts with various
    evaluation metrics for assessing model performance.
    """
    
    @staticmethod
    def calculate_rmse(y_true: Union[np.ndarray, pd.Series],
                       y_pred: Union[np.ndarray, pd.Series]) -> float:
        """
        Calculate Root Mean Squared Error (RMSE).
        
        RMSE measures the average magnitude of prediction errors,
        giving more weight to larger errors. Useful for detecting
        large prediction deviations.
        
        Formula: RMSE = sqrt(mean((y_true - y_pred)²))
        
        Args:
            y_true: Actual values (array-like)
            y_pred: Predicted values (array-like)
            
        Returns:
            float: RMSE value (lower is better, min = 0)
        """
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        logger.info(f"RMSE calculated: {rmse:.4f}")
        return rmse
    
    @staticmethod
    def calculate_mae(y_true: Union[np.ndarray, pd.Series],
                      y_pred: Union[np.ndarray, pd.Series]) -> float:
        """
        Calculate Mean Absolute Error (MAE).
        
        MAE measures the average absolute difference between actual
        and predicted values. More robust to outliers than RMSE.
        
        Formula: MAE = mean(|y_true - y_pred|)
        
        Args:
            y_true: Actual values (array-like)
            y_pred: Predicted values (array-like)
            
        Returns:
            float: MAE value (lower is better, min = 0)
        """
        mae = mean_absolute_error(y_true, y_pred)
        logger.info(f"MAE calculated: {mae:.4f}")
        return mae
    
    @staticmethod
    def calculate_r2(y_true: Union[np.ndarray, pd.Series],
                     y_pred: Union[np.ndarray, pd.Series]) -> float:
        """
        Calculate R² Score (Coefficient of Determination).
        
        R² indicates the proportion of variance in the dependent variable
        that is explained by the model.
        
        Formula: R² = 1 - (SS_res / SS_tot)
        
        Args:
            y_true: Actual values (array-like)
            y_pred: Predicted values (array-like)
            
        Returns:
            float: R² value (range: -∞ to 1, higher is better)
                  1.0 = perfect fit, 0.0 = model explains no variance
        """
        r2 = r2_score(y_true, y_pred)
        logger.info(f"R² Score calculated: {r2:.4f}")
        return r2
    
    @staticmethod
    def calculate_mape(y_true: Union[np.ndarray, pd.Series],
                       y_pred: Union[np.ndarray, pd.Series]) -> float:
        """
        Calculate Mean Absolute Percentage Error (MAPE).
        
        MAPE expresses accuracy as a percentage, making it easier to
        interpret. Useful for comparing forecasts across different scales.
        
        Formula: MAPE = mean(|y_true - y_pred| / |y_true|) * 100
        
        Args:
            y_true: Actual values (array-like)
            y_pred: Predicted values (array-like)
            
        Returns:
            float: MAPE value in percentage (lower is better)
            
        Note:
            Returns inf or nan if y_true contains zeros.
        """
        mape = mean_absolute_percentage_error(y_true, y_pred)
        logger.info(f"MAPE calculated: {mape:.4f}")
        return mape
    
    @staticmethod
    def evaluate_model(y_true: Union[np.ndarray, pd.Series],
                       y_pred: Union[np.ndarray, pd.Series],
                       include_mape: bool = True) -> Dict[str, Union[float, None]]:
        """
        Comprehensive model evaluation with all metrics.
        
        Calculates RMSE, MAE, R², and optionally MAPE in a single call.
        
        Args:
            y_true: Actual values (array-like)
            y_pred: Predicted values (array-like)
            include_mape: Whether to include MAPE (default: True)
            
        Returns:
            dict: Dictionary containing all calculated metrics
                {
                    'rmse': float,
                    'mae': float,
                    'r2': float,
                    'mape': float or None (if include_mape=True)
                }
        """
        metrics: Dict[str, Union[float, None]] = {
            'rmse': ModelEvaluator.calculate_rmse(y_true, y_pred),
            'mae': ModelEvaluator.calculate_mae(y_true, y_pred),
            'r2': ModelEvaluator.calculate_r2(y_true, y_pred),
        }
        
        if include_mape:
            try:
                metrics['mape'] = ModelEvaluator.calculate_mape(y_true, y_pred)
            except Exception as e:
                logger.warning(f"Could not calculate MAPE: {e}")
                metrics['mape'] = None
        
        logger.info(f"Model evaluation complete: {metrics}")
        return metrics
    
    @staticmethod
    def evaluate_by_horizon(y_true: pd.DataFrame,
                           y_pred: pd.DataFrame,
                           horizon_column: str = 'horizon') -> Dict[int, Dict[str, Union[float, None]]]:
        """
        Evaluate model performance across different forecast horizons.
        
        Useful for multi-step ahead forecasts where performance may vary
        by prediction distance.
        
        Args:
            y_true: DataFrame with actual values and horizon labels
            y_pred: DataFrame with predictions and horizon labels
            horizon_column: Name of column identifying forecast horizon
            
        Returns:
            dict: Metrics for each horizon level
                {
                    horizon_1: {'rmse': ..., 'mae': ..., 'r2': ...},
                    horizon_2: {...},
                    ...
                }
        """
        results = {}
        
        if horizon_column not in y_true.columns or horizon_column not in y_pred.columns:
            logger.error(f"Horizon column '{horizon_column}' not found in dataframes")
            return results
        
        horizons = y_true[horizon_column].unique()
        
        for horizon in sorted(horizons):
            mask_true = y_true[horizon_column] == horizon
            mask_pred = y_pred[horizon_column] == horizon
            
            y_t = y_true[mask_true].drop(columns=[horizon_column])
            y_p = y_pred[mask_pred].drop(columns=[horizon_column])
            
            if len(y_t) > 0 and len(y_p) > 0:
                metrics = ModelEvaluator.evaluate_model(y_t.values.flatten(), 
                                                       y_p.values.flatten())
                results[int(horizon)] = metrics
                logger.info(f"Horizon {horizon} metrics: {metrics}")
        
        return results
    
    @staticmethod
    def print_evaluation_report(metrics: Dict[str, float]) -> str:
        """
        Generate a formatted evaluation report.
        
        Args:
            metrics: Dictionary of metrics from evaluate_model()
            
        Returns:
            str: Formatted evaluation report
        """
        report = "\n" + "="*50 + "\n"
        report += "MODEL EVALUATION REPORT\n"
        report += "="*50 + "\n"
        report += f"RMSE (Root Mean Squared Error): {metrics.get('rmse', 'N/A'):.4f}\n"
        report += f"MAE  (Mean Absolute Error):     {metrics.get('mae', 'N/A'):.4f}\n"
        report += f"R²   (Coefficient of Determination): {metrics.get('r2', 'N/A'):.4f}\n"
        
        if 'mape' in metrics and metrics['mape'] is not None:
            report += f"MAPE (Mean Absolute Percentage Error): {metrics['mape']:.4f}%\n"
        
        report += "="*50 + "\n"
        
        return report
