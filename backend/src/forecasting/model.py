"""
Baseline Forecasting Model for Supply Chain Demand Prediction.

This module implements a baseline forecasting model using multiple techniques:
- Simple Moving Average (SMA)
- Exponential Smoothing
- Trend-based forecasting
- Revenue forecasting
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)


class BaselineForecaster:
    """
    Baseline forecasting model for supply chain demand prediction.
    
    Supports multiple forecasting techniques and aggregation levels:
    - Daily, weekly, monthly forecasts
    - Product-level and category-level forecasts
    - Simple Moving Average and Exponential Smoothing
    """
    
    def __init__(self, 
                 window_size: int = 7,
                 alpha: float = 0.3):
        """
        Initialize the forecaster.
        
        Args:
            window_size: Number of periods for moving average (default: 7 days)
            alpha: Smoothing factor for exponential smoothing [0, 1] (default: 0.3)
        """
        self.window_size = window_size
        self.alpha = alpha
        self.model = None
        self.scaler_mean = None
        self.scaler_std = None
        logger.info(f"BaselineForecaster initialized with window_size={window_size}, alpha={alpha}")
    
    def prepare_timeseries_data(self, 
                                df: pd.DataFrame,
                                date_col: str = "order_date",
                                value_col: str = "quantity",
                                freq: str = "D") -> pd.Series:
        """
        Prepare time series data by aggregating values by date.
        
        Args:
            df: Input dataframe
            date_col: Column containing dates
            value_col: Column containing values to forecast
            freq: Frequency for resampling ('D'=daily, 'W'=weekly, 'M'=monthly)
        
        Returns:
            Time series as pandas Series with DatetimeIndex
        """
        df_copy = df.copy()
        df_copy[date_col] = pd.to_datetime(df_copy[date_col])
        
        # Aggregate by date
        ts = df_copy.groupby(date_col)[value_col].sum()
        
        # Resample to specified frequency and forward-fill missing dates
        ts = ts.asfreq(freq, fill_value=0)
        
        logger.info(f"Prepared time series: {len(ts)} periods, freq={freq}")
        return ts
    
    def simple_moving_average(self, 
                              ts: pd.Series,
                              periods: int) -> pd.Series:
        """
        Calculate Simple Moving Average (SMA).
        
        Args:
            ts: Time series data
            periods: Number of periods for moving average
        
        Returns:
            Series containing SMA values
        """
        sma = ts.rolling(window=periods).mean()
        logger.info(f"Calculated SMA with window={periods}")
        return sma
    
    def exponential_smoothing(self, 
                             ts: pd.Series,
                             alpha: Optional[float] = None) -> pd.Series:
        """
        Calculate Exponential Smoothing forecast.
        
        Args:
            ts: Time series data
            alpha: Smoothing factor [0, 1]. If None, uses self.alpha
        
        Returns:
            Series containing exponentially smoothed values
        """
        if alpha is None:
            alpha = self.alpha
        
        es = ts.ewm(span=1/alpha, adjust=False).mean()
        logger.info(f"Calculated exponential smoothing with alpha={alpha}")
        return es
    
    def forecast_sma(self, 
                     ts: pd.Series,
                     periods: int) -> np.ndarray:
        """
        Forecast future values using Simple Moving Average.
        
        Args:
            ts: Historical time series data
            periods: Number of periods to forecast ahead
        
        Returns:
            Array of forecasted values
        """
        sma_value = ts.iloc[-self.window_size:].mean()
        forecast = np.full(periods, sma_value)
        logger.info(f"SMA forecast for {periods} periods: {sma_value:.2f}")
        return forecast
    
    def forecast_exponential_smoothing(self, 
                                       ts: pd.Series,
                                       periods: int) -> np.ndarray:
        """
        Forecast future values using Exponential Smoothing.
        
        Args:
            ts: Historical time series data
            periods: Number of periods to forecast ahead
        
        Returns:
            Array of forecasted values
        """
        # Use the last smoothed value as the forecast
        es = self.exponential_smoothing(ts)
        forecast_value = es.iloc[-1]
        forecast = np.full(periods, forecast_value)
        logger.info(f"Exponential smoothing forecast for {periods} periods: {forecast_value:.2f}")
        return forecast
    
    def forecast_trend(self, 
                      ts: pd.Series,
                      periods: int) -> np.ndarray:
        """
        Forecast future values using linear trend.
        
        Args:
            ts: Historical time series data
            periods: Number of periods to forecast ahead
        
        Returns:
            Array of forecasted values
        """
        # Prepare data for linear regression
        X = np.arange(len(ts)).reshape(-1, 1)
        y = ts.values
        
        # Fit linear model
        model = LinearRegression()
        model.fit(X, y)
        
        # Forecast future periods
        future_X = np.arange(len(ts), len(ts) + periods).reshape(-1, 1)
        forecast = model.predict(future_X)
        
        # Ensure no negative values for quantity
        forecast = np.maximum(forecast, 0)
        
        logger.info(f"Trend forecast for {periods} periods, slope={model.coef_[0]:.4f}")
        return forecast
    
    def fit(self, 
            df: pd.DataFrame,
            date_col: str = "order_date",
            value_col: str = "quantity",
            freq: str = "D") -> Dict:
        """
        Fit the baseline model on historical data.
        
        Args:
            df: Training dataframe
            date_col: Column containing dates
            value_col: Column containing values to forecast
            freq: Frequency for resampling
        
        Returns:
            Dictionary with model metrics and statistics
        """
        # Prepare time series
        ts = self.prepare_timeseries_data(df, date_col, value_col, freq)
        
        # Calculate statistics
        stats = {
            "mean": float(ts.mean()),
            "std": float(ts.std()),
            "min": float(ts.min()),
            "max": float(ts.max()),
            "periods": len(ts),
            "date_range": f"{ts.index.min()} to {ts.index.max()}"
        }
        
        # Store for later use
        self.scaler_mean = stats["mean"]
        self.scaler_std = stats["std"]
        self.model = ts
        
        logger.info(f"Model fitted. Stats: mean={stats['mean']:.2f}, std={stats['std']:.2f}")
        return stats
    
    def predict(self, 
                df: pd.DataFrame,
                periods: int = 7,
                date_col: str = "order_date",
                value_col: str = "quantity",
                freq: str = "D",
                methods: Optional[List[str]] = None) -> Dict:
        """
        Generate forecasts using multiple methods.
        
        Args:
            df: Input dataframe
            periods: Number of periods to forecast
            date_col: Column containing dates
            value_col: Column containing values
            freq: Frequency for resampling
            methods: List of methods to use ('sma', 'es', 'trend'). 
                    If None, uses all methods.
        
        Returns:
            Dictionary containing forecasts from different methods
        """
        if methods is None:
            methods = ['sma', 'es', 'trend']
        
        # Prepare time series
        ts = self.prepare_timeseries_data(df, date_col, value_col, freq)
        
        # Generate forecasts
        forecasts = {}
        
        if 'sma' in methods:
            forecasts['sma'] = self.forecast_sma(ts, periods)
        
        if 'es' in methods:
            forecasts['exponential_smoothing'] = self.forecast_exponential_smoothing(ts, periods)
        
        if 'trend' in methods:
            forecasts['trend'] = self.forecast_trend(ts, periods)
        
        # Ensemble forecast (average of all methods)
        if len(forecasts) > 1:
            forecasts['ensemble'] = np.mean(list(forecasts.values()), axis=0)
        
        return forecasts
    
    def forecast_by_category(self, 
                            df: pd.DataFrame,
                            periods: int = 7,
                            category_col: str = "category",
                            date_col: str = "order_date",
                            value_col: str = "quantity") -> Dict[str, Dict]:
        """
        Generate category-level forecasts.
        
        Args:
            df: Input dataframe
            periods: Number of periods to forecast
            category_col: Column containing category names
            date_col: Column containing dates
            value_col: Column containing values
        
        Returns:
            Dictionary with forecasts for each category
        """
        category_forecasts = {}
        
        for category in df[category_col].unique():
            category_df = df[df[category_col] == category]
            forecasts = self.predict(
                category_df,
                periods=periods,
                date_col=date_col,
                value_col=value_col
            )
            category_forecasts[category] = forecasts
            logger.info(f"Generated forecasts for category: {category}")
        
        return category_forecasts
    
    def forecast_revenue(self, 
                        df: pd.DataFrame,
                        periods: int = 7,
                        date_col: str = "order_date") -> Dict:
        """
        Forecast revenue (quantity * unit_price).
        
        Args:
            df: Input dataframe
            periods: Number of periods to forecast
            date_col: Column containing dates
        
        Returns:
            Dictionary with revenue forecasts
        """
        df_copy = df.copy()
        df_copy['revenue'] = df_copy['quantity'] * df_copy['unit_price'] * (1 - df_copy.get('discount', 0))
        
        forecasts = self.predict(
            df_copy,
            periods=periods,
            date_col=date_col,
            value_col='revenue',
            freq='D'
        )
        
        logger.info("Generated revenue forecasts")
        return forecasts
    
    def evaluate(self, 
                 df: pd.DataFrame,
                 train_test_split: float = 0.8,
                 date_col: str = "order_date",
                 value_col: str = "quantity",
                 freq: str = "D") -> Dict:
        """
        Evaluate model performance using train-test split.
        
        Args:
            df: Input dataframe
            train_test_split: Fraction of data to use for training
            date_col: Column containing dates
            value_col: Column containing values
            freq: Frequency for resampling
        
        Returns:
            Dictionary with evaluation metrics
        """
        ts = self.prepare_timeseries_data(df, date_col, value_col, freq)
        
        # Split data
        split_idx = int(len(ts) * train_test_split)
        train_ts = ts.iloc[:split_idx]
        test_ts = ts.iloc[split_idx:]
        
        # Generate forecasts for test period
        forecasts = self.predict(
            df.iloc[:int(len(df) * train_test_split)],
            periods=len(test_ts),
            date_col=date_col,
            value_col=value_col,
            freq=freq
        )
        
        # Calculate metrics for each method
        metrics = {}
        test_values = test_ts.values
        
        for method_name, forecast_values in forecasts.items():
            mae = mean_absolute_error(test_values, forecast_values[:len(test_values)])
            rmse = np.sqrt(mean_squared_error(test_values, forecast_values[:len(test_values)]))
            
            # R² score (handle edge cases)
            try:
                r2 = r2_score(test_values, forecast_values[:len(test_values)])
            except:
                r2 = np.nan
            
            metrics[method_name] = {
                'mae': float(mae),
                'rmse': float(rmse),
                'r2': float(r2) if not np.isnan(r2) else None,
                'mape': float(np.mean(np.abs((test_values - forecast_values[:len(test_values)]) / (test_values + 1e-8)))) * 100
            }
        
        logger.info(f"Model evaluation complete. Metrics: {metrics}")
        return metrics


def create_forecast_summary(df: pd.DataFrame,
                          periods: int = 7) -> Dict:
    """
    Create a complete forecast summary.
    
    Args:
        df: Input dataframe
        periods: Number of periods to forecast
    
    Returns:
        Dictionary with comprehensive forecast information
    """
    forecaster = BaselineForecaster()
    
    # Fit model
    fit_stats = forecaster.fit(df)
    
    # Generate forecasts
    quantity_forecasts = forecaster.predict(df, periods=periods)
    revenue_forecasts = forecaster.forecast_revenue(df, periods=periods)
    category_forecasts = forecaster.forecast_by_category(df, periods=periods)
    
    # Evaluate model
    metrics = forecaster.evaluate(df)
    
    summary = {
        'fit_statistics': fit_stats,
        'quantity_forecast': {k: v.tolist() for k, v in quantity_forecasts.items()},
        'revenue_forecast': {k: v.tolist() for k, v in revenue_forecasts.items()},
        'category_forecast': {
            cat: {method: val.tolist() for method, val in forecasts.items()} 
            for cat, forecasts in category_forecasts.items()
        },
        'metrics': metrics,
        'forecast_periods': periods,
        'timestamp': datetime.now().isoformat()
    }
    
    return summary
