"""
Random Forest forecasting model for supply chain demand prediction.

This module implements a Random Forest model for multivariate time series
forecasting of demand, incorporating temporal, lag, and aggregation features.
"""
import logging
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error
)

logger = logging.getLogger(__name__)

# Type aliases
FloatArray = np.ndarray
PredictionTuple = Tuple[np.ndarray, np.ndarray, np.ndarray]


class RandomForestForecaster:
    """
    Random Forest model for supply chain demand forecasting.
    
    Features:
    - Multi-step ahead forecasting
    - Feature importance analysis
    - Hyperparameter tuning support
    - Model persistence (save/load)
    - Performance metrics calculation
    """
    
    def __init__(self,
                 n_estimators: int = 100,
                 max_depth: Optional[int] = 15,
                 min_samples_split: int = 5,
                 min_samples_leaf: int = 2,
                 random_state: int = 42,
                 n_jobs: int = -1):
        """
        Initialize Random Forest Forecaster.
        
        Args:
            n_estimators: Number of trees in the forest (default: 100)
            max_depth: Maximum depth of trees (default: 15)
            min_samples_split: Minimum samples to split node (default: 5)
            min_samples_leaf: Minimum samples in leaf node (default: 2)
            random_state: Random seed for reproducibility (default: 42)
            n_jobs: Number of parallel jobs (-1 = all processors)
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.n_jobs = n_jobs
        
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.training_history = {}
        
        logger.info(
            f"RandomForestForecaster initialized with n_estimators={n_estimators}, "
            f"max_depth={max_depth}"
        )
    
    def _create_model(self) -> RandomForestRegressor:
        """Create and return a new Random Forest model."""
        return RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            warm_start=False
        )
    
    def prepare_data(self,
                    X: pd.DataFrame,
                    y: pd.Series,
                    test_size: float = 0.2,
                    scale: bool = True) -> Tuple[np.ndarray, np.ndarray, 
                                                 np.ndarray, np.ndarray]:
        """
        Prepare and split data for training.
        
        Args:
            X: Feature matrix
            y: Target variable
            test_size: Proportion of data for testing (default: 0.2)
            scale: Whether to scale features (default: True)
        
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        # Remove rows with NaN values
        valid_idx = ~(X.isna().any(axis=1) | y.isna())
        X_clean = X[valid_idx].reset_index(drop=True)
        y_clean = y[valid_idx].reset_index(drop=True)
        
        logger.info(f"Cleaned data: {len(X_clean)} samples from {len(X)}")
        
        # Store feature names
        self.feature_names = X_clean.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_clean, y_clean,
            test_size=test_size,
            random_state=self.random_state
        )
        
        # Scale features if requested
        if scale:
            self.scaler = StandardScaler()
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)
        else:
            X_train = X_train.values
            X_test = X_test.values
        
        logger.info(
            f"Data split: {len(X_train)} training, {len(X_test)} testing samples"
        )
        
        return X_train, X_test, y_train.values, y_test.values
    
    def train(self,
             X: pd.DataFrame,
             y: pd.Series,
             test_size: float = 0.2,
             scale: bool = True,
             verbose: bool = True) -> Dict[str, float]:
        """
        Train the Random Forest model.
        
        Args:
            X: Feature matrix
            y: Target variable
            test_size: Test/train split ratio (default: 0.2)
            scale: Whether to scale features (default: True)
            verbose: Print training metrics (default: True)
        
        Returns:
            Dictionary with training and testing metrics
        """
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data(
            X, y, test_size=test_size, scale=scale
        )
        
        # Create and train model
        self.model = self._create_model()
        self.model.fit(X_train, y_train)
        
        logger.info("Random Forest model training completed")
        
        # Evaluate on training and test sets
        metrics = self.evaluate(X_train, y_train, X_test, y_test)
        
        # Store in history
        self.training_history = metrics
        
        if verbose:
            logger.info(f"Training Metrics - MAE: {metrics['train_mae']:.4f}, "
                       f"RMSE: {metrics['train_rmse']:.4f}, "
                       f"R²: {metrics['train_r2']:.4f}")
            logger.info(f"Testing Metrics - MAE: {metrics['test_mae']:.4f}, "
                       f"RMSE: {metrics['test_rmse']:.4f}, "
                       f"R²: {metrics['test_r2']:.4f}")
        
        return metrics
    
    def evaluate(self,
                X_train: np.ndarray,
                y_train: np.ndarray,
                X_test: np.ndarray,
                y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance on training and test sets.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target
        
        Returns:
            Dictionary with metrics: MAE, RMSE, R², MAPE
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Predictions
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        
        # Training metrics
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        train_r2 = r2_score(y_train, y_train_pred)
        train_mape = mean_absolute_percentage_error(y_train, y_train_pred)
        
        # Test metrics
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_r2 = r2_score(y_test, y_test_pred)
        test_mape = mean_absolute_percentage_error(y_test, y_test_pred)
        
        metrics = {
            'train_mae': train_mae,
            'train_rmse': train_rmse,
            'train_r2': train_r2,
            'train_mape': train_mape,
            'test_mae': test_mae,
            'test_rmse': test_rmse,
            'test_r2': test_r2,
            'test_mape': test_mape
        }
        
        return metrics
    
    def predict(self,
               X: pd.DataFrame,
               scale: bool = True) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Feature matrix
            scale: Whether to scale features (default: True)
        
        Returns:
            Array of predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        X_data = X.values
        
        if scale and self.scaler is not None:
            X_data = self.scaler.transform(X_data)
        
        predictions = self.model.predict(X_data)
        
        logger.info(f"Generated {len(predictions)} predictions")
        
        return predictions
    
    def predict_with_confidence(self,
                               X: pd.DataFrame,
                               scale: bool = True,
                               percentile: float = 0.16) -> Tuple[np.ndarray, 
                                                                  np.ndarray, 
                                                                  np.ndarray]:
        """
        Make predictions with confidence intervals using tree variance.
        
        Args:
            X: Feature matrix
            scale: Whether to scale features (default: True)
            percentile: Percentile for confidence interval bounds (default: 0.16)
        
        Returns:
            Tuple of (predictions, lower_bound, upper_bound)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        X_data = X.values
        
        if scale and self.scaler is not None:
            X_data = self.scaler.transform(X_data)
        
        # Get predictions from individual trees
        predictions_list = np.array([
            tree.predict(X_data) for tree in self.model.estimators_
        ])
        
        # Mean prediction
        mean_pred = predictions_list.mean(axis=0)
        
        # Confidence intervals
        lower_bound = np.percentile(predictions_list, percentile, axis=0)
        upper_bound = np.percentile(predictions_list, 100 - percentile, axis=0)
        
        logger.info(f"Generated {len(mean_pred)} predictions with confidence intervals")
        
        return mean_pred, lower_bound, upper_bound
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance scores.
        
        Args:
            top_n: Number of top features to return (default: 20)
        
        Returns:
            DataFrame with feature names and importance scores
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        if self.feature_names is None:
            raise ValueError("Feature names not stored. Retrain model.")
        
        importances = self.model.feature_importances_
        
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        logger.info(f"Top {top_n} features: {feature_importance_df.head(top_n)['feature'].tolist()}")
        
        return feature_importance_df.head(top_n)
    
    def get_training_history(self) -> Dict[str, float]:
        """Get training metrics history."""
        return self.training_history.copy()
    
    def save_model(self, filepath: str) -> None:
        """
        Save trained model to disk using joblib.
        
        Args:
            filepath: Path to save model (.joblib or .pkl)
        """
        import joblib
        
        if self.model is None:
            raise ValueError("No trained model to save.")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'hyperparameters': {
                'n_estimators': self.n_estimators,
                'max_depth': self.max_depth,
                'min_samples_split': self.min_samples_split,
                'min_samples_leaf': self.min_samples_leaf,
                'random_state': self.random_state
            }
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load trained model from disk.
        
        Args:
            filepath: Path to load model from
        """
        import joblib
        
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        
        hyperparams = model_data['hyperparameters']
        self.n_estimators = hyperparams['n_estimators']
        self.max_depth = hyperparams['max_depth']
        self.min_samples_split = hyperparams['min_samples_split']
        self.min_samples_leaf = hyperparams['min_samples_leaf']
        self.random_state = hyperparams['random_state']
        
        logger.info(f"Model loaded from {filepath}")
    
    def get_model_info(self) -> Dict:
        """Get model information and hyperparameters."""
        return {
            'model_type': 'RandomForest',
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'random_state': self.random_state,
            'feature_count': len(self.feature_names) if self.feature_names else 0,
            'is_trained': self.model is not None,
            'has_scaler': self.scaler is not None,
            'training_metrics': self.training_history
        }
