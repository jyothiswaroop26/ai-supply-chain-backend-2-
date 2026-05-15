"""
Example usage of model saving and loading functionality.

This script demonstrates how to:
1. Train models
2. Save them with metadata
3. Load and use saved models
4. Manage model collections
"""

import pandas as pd
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_save_baseline_model():
    """Example: Train and save a BaselineForecaster model."""
    from backend.src.forecasting import BaselineForecaster, get_model_manager
    
    logger.info("=" * 60)
    logger.info("Example 1: Save BaselineForecaster Model")
    logger.info("=" * 60)
    
    # In a real scenario, load actual data
    # df = pd.read_csv("backend/data/sales_data.csv")
    
    # Create sample data for demonstration
    df = pd.DataFrame({
        'order_date': pd.date_range('2024-01-01', periods=100, freq='D'),
        'quantity': [100 + i % 20 for i in range(100)],
        'category': ['A', 'B', 'C'] * 33 + ['A']
    })
    
    # Train the model
    logger.info("Training BaselineForecaster...")
    forecaster = BaselineForecaster(window_size=7, alpha=0.3)
    fit_stats = forecaster.fit(df)
    logger.info(f"Fit statistics: {fit_stats}")
    
    # Evaluate model
    metrics = forecaster.evaluate(df)
    logger.info(f"Model metrics: {metrics}")
    
    # Save model using ModelManager
    logger.info("Saving model...")
    manager = get_model_manager()
    
    model_path = manager.save_baseline_model(
        model=forecaster,
        model_name="baseline_demand_v1",
        metadata={
            "description": "Baseline demand forecast model",
            "window_size": 7,
            "alpha": 0.3,
            "evaluation_date": datetime.now().isoformat(),
            "metrics": metrics
        }
    )
    
    logger.info(f"Model saved successfully to: {model_path}")
    
    # List all saved models
    logger.info("Listing all saved models...")
    all_models = manager.list_models()
    logger.info(f"Saved models: {all_models}")
    
    return manager


def example_load_and_use_baseline_model(manager):
    """Example: Load and use a saved BaselineForecaster model."""
    from backend.src.forecasting import get_model_manager
    
    logger.info("=" * 60)
    logger.info("Example 2: Load and Use BaselineForecaster Model")
    logger.info("=" * 60)
    
    if manager is None:
        manager = get_model_manager()
    
    # Load the model
    logger.info("Loading baseline model...")
    forecaster = manager.load_baseline_model("baseline_demand_v1")
    
    # Get model information
    model_info = forecaster.get_model_info()
    logger.info(f"Model information: {model_info}")
    
    # Create sample data for prediction
    df_new = pd.DataFrame({
        'order_date': pd.date_range('2024-05-01', periods=30, freq='D'),
        'quantity': [110 + i % 15 for i in range(30)],
        'category': ['A', 'B', 'C'] * 10
    })
    
    # Generate predictions
    logger.info("Generating predictions...")
    predictions = forecaster.predict(df_new, periods=7)
    logger.info(f"Predictions: {predictions}")
    
    # Get metadata
    metadata = manager.get_model_metadata("baseline_demand_v1", model_type="baseline")
    logger.info(f"Model metadata: {metadata}")


def example_save_random_forest_model():
    """Example: Train and save a RandomForestForecaster model."""
    from backend.src.forecasting.models import RandomForestForecaster
    from backend.src.forecasting import get_model_manager
    from sklearn.datasets import make_regression
    
    logger.info("=" * 60)
    logger.info("Example 3: Save RandomForestForecaster Model")
    logger.info("=" * 60)
    
    # Create sample data
    logger.info("Creating sample training data...")
    X, y = make_regression(n_samples=200, n_features=10, random_state=42)[:2]
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
    y = pd.Series(y, name='target')
    
    # Train the model
    logger.info("Training RandomForestForecaster...")
    rf_model = RandomForestForecaster(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5
    )
    metrics = rf_model.train(X, y, test_size=0.2)
    logger.info(f"Training metrics: {metrics}")
    
    # Get feature importance
    feature_importance = rf_model.get_feature_importance(top_n=10)
    logger.info(f"Top features:\n{feature_importance}")
    
    # Save model using ModelManager
    logger.info("Saving RandomForest model...")
    manager = get_model_manager()
    
    model_path = manager.save_random_forest_model(
        model=rf_model,
        model_name="rf_demand_v1",
        metadata={
            "description": "Random Forest demand forecast",
            "n_estimators": 100,
            "max_depth": 15,
            "training_date": datetime.now().isoformat(),
            "features": X.columns.tolist()
        }
    )
    
    logger.info(f"Model saved successfully to: {model_path}")


def example_load_and_use_random_forest_model():
    """Example: Load and use a saved RandomForestForecaster model."""
    from backend.src.forecasting import get_model_manager
    import numpy as np
    
    logger.info("=" * 60)
    logger.info("Example 4: Load and Use RandomForestForecaster Model")
    logger.info("=" * 60)
    
    # Load the model
    logger.info("Loading RandomForest model...")
    manager = get_model_manager()
    rf_model = manager.load_random_forest_model("rf_demand_v1")
    
    # Get model information
    model_info = rf_model.get_model_info()
    logger.info(f"Model information: {model_info}")
    
    # Create sample data for prediction
    X_new = pd.DataFrame(
        np.random.randn(10, 10),
        columns=[f'feature_{i}' for i in range(10)]
    )
    
    # Generate predictions
    logger.info("Generating predictions...")
    predictions = rf_model.predict(X_new)
    logger.info(f"Predictions shape: {predictions.shape}")
    logger.info(f"Sample predictions: {predictions[:3]}")
    
    # Generate predictions with confidence intervals
    logger.info("Generating predictions with confidence intervals...")
    mean_pred, lower, upper = rf_model.predict_with_confidence(X_new)
    logger.info(f"Mean prediction: {mean_pred[:3]}")
    logger.info(f"Confidence interval width: {upper[:3] - lower[:3]}")


def example_model_management():
    """Example: Manage model collection."""
    from backend.src.forecasting import get_model_manager
    
    logger.info("=" * 60)
    logger.info("Example 5: Model Management")
    logger.info("=" * 60)
    
    manager = get_model_manager()
    
    # List all models
    logger.info("Listing all saved models...")
    all_models = manager.list_models()
    logger.info(f"Available models: {all_models}")
    
    # Export all model information
    logger.info("Exporting model registry...")
    registry = manager.export_model_info()
    logger.info(f"Model registry keys: {list(registry.keys())}")
    
    # Get metadata for specific model
    try:
        metadata = manager.get_model_metadata("baseline_demand_v1", model_type="baseline")
        logger.info(f"Baseline model metadata: {metadata}")
    except Exception as e:
        logger.info(f"Note: Model not found or not yet saved: {e}")
    
    # Note: Uncomment to test model deletion
    # logger.info("Deleting old model...")
    # manager.delete_model("old_model_v1", model_type="baseline")
    # logger.info("Model deleted successfully")


def main():
    """Run all examples."""
    logger.info("\n" + "=" * 60)
    logger.info("MODEL PERSISTENCE AND MANAGEMENT EXAMPLES")
    logger.info("=" * 60 + "\n")
    
    # Example 1: Save baseline model
    manager = example_save_baseline_model()
    
    # Example 2: Load and use baseline model
    example_load_and_use_baseline_model(manager)
    
    # Example 3: Save RandomForest model
    example_save_random_forest_model()
    
    # Example 4: Load and use RandomForest model
    example_load_and_use_random_forest_model()
    
    # Example 5: Manage models
    example_model_management()
    
    logger.info("\n" + "=" * 60)
    logger.info("ALL EXAMPLES COMPLETED SUCCESSFULLY")
    logger.info("=" * 60 + "\n")


if __name__ == "__main__":
    main()
