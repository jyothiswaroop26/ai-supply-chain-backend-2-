"""
Feature engineering utilities for supply chain demand forecasting.

This module provides comprehensive feature engineering functions for creating
predictive features from sales data, including:
- Temporal features (seasonality, cyclical patterns)
- Lag features (historical demand)
- Rolling statistics (trends, volatility)
- Category and regional aggregations
- Price and revenue features
"""
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureEngineering:
    """
    Feature engineering class for supply chain demand forecasting.
    
    Transforms raw sales data into predictive features through:
    - Temporal decomposition
    - Statistical aggregations
    - Lag-based features
    - Category and regional grouping
    """
    
    def __init__(self, data_df: pd.DataFrame):
        """
        Initialize feature engineering with data.
        
        Args:
            data_df: DataFrame with preprocessed sales data
        """
        self.df = data_df.copy()
        self.features_df = pd.DataFrame()
        logger.info(f"FeatureEngineering initialized with {len(self.df)} rows")
    
    def create_temporal_features(self, date_col: str = "order_date") -> pd.DataFrame:
        """
        Create temporal features from date column.
        
        Features created:
        - day_of_week: 0-6 (Monday-Sunday)
        - day_of_month: 1-31
        - week_of_year: 1-52
        - month: 1-12
        - quarter: 1-4
        - is_weekend: Binary indicator
        - is_month_end: Binary indicator
        - day_of_year: 1-365
        
        Args:
            date_col: Name of date column
        
        Returns:
            DataFrame with temporal features
        """
        df = self.df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        
        temporal_features = pd.DataFrame(index=df.index)
        temporal_features['day_of_week'] = df[date_col].dt.dayofweek
        temporal_features['day_of_month'] = df[date_col].dt.day
        temporal_features['week_of_year'] = df[date_col].dt.isocalendar().week
        temporal_features['month'] = df[date_col].dt.month
        temporal_features['quarter'] = df[date_col].dt.quarter
        temporal_features['is_weekend'] = (df[date_col].dt.dayofweek >= 5).astype(int)
        temporal_features['is_month_end'] = df[date_col].dt.is_month_end.astype(int)
        temporal_features['day_of_year'] = df[date_col].dt.dayofyear
        
        # Cyclical encoding for periodicity
        temporal_features['sin_month'] = np.sin(2 * np.pi * temporal_features['month'] / 12)
        temporal_features['cos_month'] = np.cos(2 * np.pi * temporal_features['month'] / 12)
        temporal_features['sin_day_of_week'] = np.sin(2 * np.pi * temporal_features['day_of_week'] / 7)
        temporal_features['cos_day_of_week'] = np.cos(2 * np.pi * temporal_features['day_of_week'] / 7)
        
        self.features_df = pd.concat([self.features_df, temporal_features], axis=1)
        logger.info(f"Created {len(temporal_features.columns)} temporal features")
        
        return temporal_features
    
    def create_lag_features(self, 
                           value_col: str = "quantity",
                           lags: List[int] = None,
                           group_cols: List[str] = None) -> pd.DataFrame:
        """
        Create lagged features for capturing temporal dependencies.
        
        Args:
            value_col: Column to create lags for (default: quantity)
            lags: List of lag periods (default: [1, 7, 30])
            group_cols: Columns to group by (e.g., ['product_id', 'region'])
        
        Returns:
            DataFrame with lag features
        """
        if lags is None:
            lags = [1, 7, 14, 30]
        
        df = self.df.copy()
        df['order_date'] = pd.to_datetime(df['order_date'])
        df = df.sort_values('order_date').reset_index(drop=True)
        
        lag_features = pd.DataFrame(index=df.index)
        
        if group_cols:
            for group_col in group_cols:
                for lag in lags:
                    lag_name = f'{value_col}_lag_{lag}_{group_col}'
                    lag_features[lag_name] = (
                        df.groupby(group_col)[value_col]
                        .shift(lag)
                    )
        else:
            for lag in lags:
                lag_name = f'{value_col}_lag_{lag}'
                lag_features[lag_name] = df[value_col].shift(lag)
        
        self.features_df = pd.concat([self.features_df, lag_features], axis=1)
        logger.info(f"Created {len(lag_features.columns)} lag features")
        
        return lag_features
    
    def create_rolling_statistics(self,
                                  value_col: str = "quantity",
                                  windows: List[int] = None,
                                  group_cols: List[str] = None) -> pd.DataFrame:
        """
        Create rolling statistical features.
        
        Features created for each window:
        - rolling_mean: Average value
        - rolling_std: Standard deviation
        - rolling_min: Minimum value
        - rolling_max: Maximum value
        - rolling_sum: Total value
        
        Args:
            value_col: Column to compute statistics on
            windows: List of window sizes in days (default: [7, 30])
            group_cols: Columns to group by
        
        Returns:
            DataFrame with rolling features
        """
        if windows is None:
            windows = [7, 14, 30]
        
        df = self.df.copy()
        df['order_date'] = pd.to_datetime(df['order_date'])
        df = df.sort_values('order_date').reset_index(drop=True)
        
        rolling_features = pd.DataFrame(index=df.index)
        
        if group_cols:
            for group_col in group_cols:
                grouped = df.groupby(group_col)[value_col]
                for window in windows:
                    rolling_features[f'{value_col}_rolling_mean_{window}_{group_col}'] = grouped.transform(
                        lambda x: x.rolling(window=window, min_periods=1).mean()
                    )
                    rolling_features[f'{value_col}_rolling_std_{window}_{group_col}'] = grouped.transform(
                        lambda x: x.rolling(window=window, min_periods=1).std()
                    )
                    rolling_features[f'{value_col}_rolling_max_{window}_{group_col}'] = grouped.transform(
                        lambda x: x.rolling(window=window, min_periods=1).max()
                    )
        else:
            for window in windows:
                rolling_features[f'{value_col}_rolling_mean_{window}'] = (
                    df[value_col].rolling(window=window, min_periods=1).mean()
                )
                rolling_features[f'{value_col}_rolling_std_{window}'] = (
                    df[value_col].rolling(window=window, min_periods=1).std()
                )
                rolling_features[f'{value_col}_rolling_max_{window}'] = (
                    df[value_col].rolling(window=window, min_periods=1).max()
                )
        
        self.features_df = pd.concat([self.features_df, rolling_features], axis=1)
        logger.info(f"Created {len(rolling_features.columns)} rolling statistical features")
        
        return rolling_features
    
    def create_exponential_features(self,
                                   value_col: str = "quantity",
                                   spans: List[int] = None,
                                   group_cols: List[str] = None) -> pd.DataFrame:
        """
        Create exponentially weighted moving average (EWMA) features.
        
        EWMA gives more weight to recent observations.
        
        Args:
            value_col: Column to compute EWMA on
            spans: List of span values (default: [7, 30])
            group_cols: Columns to group by
        
        Returns:
            DataFrame with exponential features
        """
        if spans is None:
            spans = [7, 30]
        
        df = self.df.copy()
        df['order_date'] = pd.to_datetime(df['order_date'])
        df = df.sort_values('order_date').reset_index(drop=True)
        
        ewma_features = pd.DataFrame(index=df.index)
        
        if group_cols:
            for group_col in group_cols:
                grouped = df.groupby(group_col)[value_col]
                for span in spans:
                    ewma_features[f'{value_col}_ewma_{span}_{group_col}'] = grouped.transform(
                        lambda x: x.ewm(span=span, adjust=False).mean()
                    )
        else:
            for span in spans:
                ewma_features[f'{value_col}_ewma_{span}'] = (
                    df[value_col].ewm(span=span, adjust=False).mean()
                )
        
        self.features_df = pd.concat([self.features_df, ewma_features], axis=1)
        logger.info(f"Created {len(ewma_features.columns)} exponential features")
        
        return ewma_features
    
    def create_aggregation_features(self) -> pd.DataFrame:
        """
        Create aggregation features by category and region.
        
        Features include:
        - Category-level quantity totals and averages
        - Region-level quantity totals and averages
        - Product-level revenue metrics
        - Supplier-level performance metrics
        
        Returns:
            DataFrame with aggregation features
        """
        df = self.df.copy()
        df['revenue'] = df['quantity'] * df['unit_price'] * (1 - df['discount'])
        
        agg_features = pd.DataFrame(index=df.index)
        
        # Category aggregations
        if 'category' in df.columns:
            category_qty = df.groupby('category')['quantity'].transform(['sum', 'mean'])
            category_qty.columns = [f'category_qty_{col}' for col in category_qty.columns]
            agg_features = pd.concat([agg_features, category_qty], axis=1)
            
            category_rev = df.groupby('category')['revenue'].transform(['sum', 'mean'])
            category_rev.columns = [f'category_revenue_{col}' for col in category_rev.columns]
            agg_features = pd.concat([agg_features, category_rev], axis=1)
        
        # Region aggregations
        if 'region' in df.columns:
            region_qty = df.groupby('region')['quantity'].transform(['sum', 'mean'])
            region_qty.columns = [f'region_qty_{col}' for col in region_qty.columns]
            agg_features = pd.concat([agg_features, region_qty], axis=1)
            
            region_rev = df.groupby('region')['revenue'].transform(['sum', 'mean'])
            region_rev.columns = [f'region_revenue_{col}' for col in region_rev.columns]
            agg_features = pd.concat([agg_features, region_rev], axis=1)
        
        # Supplier aggregations
        if 'supplier_id' in df.columns:
            supplier_qty = df.groupby('supplier_id')['quantity'].transform('mean')
            agg_features['supplier_avg_qty'] = supplier_qty
        
        self.features_df = pd.concat([self.features_df, agg_features], axis=1)
        logger.info(f"Created {len(agg_features.columns)} aggregation features")
        
        return agg_features
    
    def create_price_features(self) -> pd.DataFrame:
        """
        Create price and discount related features.
        
        Features:
        - unit_price normalized and scaled
        - discount rate and indicator
        - revenue per unit
        - price changes
        
        Returns:
            DataFrame with price features
        """
        df = self.df.copy()
        
        price_features = pd.DataFrame(index=df.index)
        
        if 'unit_price' in df.columns:
            price_features['unit_price_normalized'] = (
                (df['unit_price'] - df['unit_price'].mean()) / (df['unit_price'].std() + 1e-8)
            )
            price_features['unit_price_log'] = np.log1p(df['unit_price'])
        
        if 'discount' in df.columns:
            price_features['has_discount'] = (df['discount'] > 0).astype(int)
            price_features['discount_rate'] = df['discount']
        
        if 'quantity' in df.columns and 'unit_price' in df.columns:
            price_features['revenue_per_unit'] = (
                df['unit_price'] * df['quantity']
            )
        
        self.features_df = pd.concat([self.features_df, price_features], axis=1)
        logger.info(f"Created {len(price_features.columns)} price features")
        
        return price_features
    
    def create_interaction_features(self) -> pd.DataFrame:
        """
        Create interaction features between key variables.
        
        Features:
        - Quantity × Price
        - Quantity × Discount
        - Category × Region interactions
        
        Returns:
            DataFrame with interaction features
        """
        df = self.df.copy()
        
        interaction_features = pd.DataFrame(index=df.index)
        
        if 'quantity' in df.columns and 'unit_price' in df.columns:
            interaction_features['qty_price_interaction'] = (
                df['quantity'] * df['unit_price']
            )
        
        if 'quantity' in df.columns and 'discount' in df.columns:
            interaction_features['qty_discount_interaction'] = (
                df['quantity'] * df['discount']
            )
        
        if 'category' in df.columns and 'region' in df.columns:
            cat_region_key = df['category'].astype(str) + '_' + df['region'].astype(str)
            interaction_features['category_region_group'] = pd.factorize(cat_region_key)[0]
        
        self.features_df = pd.concat([self.features_df, interaction_features], axis=1)
        logger.info(f"Created {len(interaction_features.columns)} interaction features")
        
        return interaction_features
    
    def create_target_variable(self, 
                              value_col: str = "quantity",
                              forecast_horizon: int = 7,
                              agg_func: str = 'sum') -> pd.Series:
        """
        Create target variable for forecasting.
        
        Args:
            value_col: Column to use as target
            forecast_horizon: Number of periods to forecast ahead
            agg_func: Aggregation function ('sum', 'mean', 'max')
        
        Returns:
            Series with target variable
        """
        df = self.df.copy()
        df['order_date'] = pd.to_datetime(df['order_date'])
        df = df.sort_values('order_date').reset_index(drop=True)
        
        # Group by date and aggregate
        ts_data = df.groupby('order_date')[value_col].agg(agg_func)
        
        # Create target as future values
        target = ts_data.shift(-forecast_horizon)
        
        logger.info(f"Created target variable with forecast horizon={forecast_horizon}")
        
        return target
    
    def engineer_all_features(self,
                             temporal: bool = True,
                             lags: bool = True,
                             rolling: bool = True,
                             ewma: bool = True,
                             aggregations: bool = True,
                             prices: bool = True,
                             interactions: bool = True) -> pd.DataFrame:
        """
        Create all feature types in sequence.
        
        Args:
            temporal: Create temporal features
            lags: Create lag features
            rolling: Create rolling statistics
            ewma: Create exponential features
            aggregations: Create aggregation features
            prices: Create price features
            interactions: Create interaction features
        
        Returns:
            DataFrame with all engineered features
        """
        self.features_df = pd.DataFrame(index=self.df.index)
        
        if temporal:
            self.create_temporal_features()
        
        if lags:
            self.create_lag_features()
        
        if rolling:
            self.create_rolling_statistics()
        
        if ewma:
            self.create_exponential_features()
        
        if aggregations:
            self.create_aggregation_features()
        
        if prices:
            self.create_price_features()
        
        if interactions:
            self.create_interaction_features()
        
        # Handle NaN values
        self.features_df = self.features_df.fillna(0)
        
        logger.info(f"Engineered {len(self.features_df.columns)} total features")
        
        return self.features_df
    
    def get_feature_names(self) -> List[str]:
        """Get list of all engineered feature names."""
        return self.features_df.columns.tolist()
    
    def get_features_dataframe(self) -> pd.DataFrame:
        """Get the engineered features DataFrame."""
        return self.features_df.copy()


def create_features_for_forecasting(data_df: pd.DataFrame,
                                   temporal: bool = True,
                                   lags: bool = True,
                                   rolling: bool = True,
                                   ewma: bool = True,
                                   aggregations: bool = True,
                                   prices: bool = True,
                                   interactions: bool = True) -> Tuple[pd.DataFrame, List[str]]:
    """
    Convenience function to create all features at once.
    
    Args:
        data_df: Raw sales data
        temporal: Include temporal features
        lags: Include lag features
        rolling: Include rolling statistics
        ewma: Include exponential features
        aggregations: Include aggregation features
        prices: Include price features
        interactions: Include interaction features
    
    Returns:
        Tuple of (features DataFrame, feature names list)
    """
    feature_engine = FeatureEngineering(data_df)
    
    features_df = feature_engine.engineer_all_features(
        temporal=temporal,
        lags=lags,
        rolling=rolling,
        ewma=ewma,
        aggregations=aggregations,
        prices=prices,
        interactions=interactions
    )
    
    feature_names = feature_engine.get_feature_names()
    
    logger.info(f"Successfully created {len(feature_names)} features for forecasting")
    
    return features_df, feature_names
