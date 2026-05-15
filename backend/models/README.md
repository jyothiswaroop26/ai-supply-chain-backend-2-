# Model Persistence and Management

This document describes the model saving and loading functionality for the Supply Chain Forecasting system.

## Overview

The system provides comprehensive model persistence capabilities for both **BaselineForecaster** and **RandomForestForecaster** models. Models are saved with metadata tracking and version management through the **ModelManager** utility.

## Directory Structure

```
backend/
├── models/                    # Central model storage
│   ├── baseline/              # Baseline models
│   │   └── {model_name}/
│   │       ├── {model_name}_model.joblib
│   │       └── metadata.json
│   ├── random_forest/         # Random Forest models
│   │   └── {model_name}/
│   │       ├── {model_name}_model.joblib
│   │       └── metadata.json
│   └── model_registry.json    # Central registry of all models
└── src/forecasting/
    ├── model.py              # BaselineForecaster with save/load
    ├── models/
    │   └── random_forest.py  # RandomForestForecaster with save/load
    └── model_manager.py      # ModelManager utility
```

## Usage

### Using ModelManager

```python
from backend.src.forecasting import ModelManager, BaselineForecaster, get_model_manager

# Get or create ModelManager (defaults to backend/models/)
manager = get_model_manager()

# Or specify custom directory
manager = ModelManager(models_dir="/custom/path/models")
```

### Saving Models

#### Save Baseline Model
```python
from backend.src.forecasting import BaselineForecaster, get_model_manager

# Train your model
forecaster = BaselineForecaster()
forecaster.fit(df)

# Save with ModelManager
manager = get_model_manager()
manager.save_baseline_model(
    model=forecaster,
    model_name="demand_forecast_v1",
    metadata={
        "description": "Monthly demand forecast",
        "data_source": "sales_data.csv"
    }
)
```

#### Save Random Forest Model
```python
from backend.src.forecasting.models import RandomForestForecaster
from backend.src.forecasting import get_model_manager

# Train your model
rf_model = RandomForestForecaster(n_estimators=100)
metrics = rf_model.train(X, y)

# Save with ModelManager
manager = get_model_manager()
manager.save_random_forest_model(
    model=rf_model,
    model_name="rf_demand_v1",
    metadata={
        "description": "Random Forest demand forecast",
        "metrics": metrics
    }
)
```

### Loading Models

#### Load Baseline Model
```python
from backend.src.forecasting import get_model_manager

manager = get_model_manager()
forecaster = manager.load_baseline_model("demand_forecast_v1")

# Use loaded model
predictions = forecaster.predict(df, periods=30)
```

#### Load Random Forest Model
```python
from backend.src.forecasting import get_model_manager

manager = get_model_manager()
rf_model = manager.load_random_forest_model("rf_demand_v1")

# Use loaded model
predictions = rf_model.predict(X)
```

### Managing Models

```python
manager = get_model_manager()

# List all saved models
models = manager.list_models()
# Returns: {'baseline': ['model1', 'model2'], 'random_forest': ['rf_model1']}

# Get model metadata
metadata = manager.get_model_metadata("demand_forecast_v1", model_type="baseline")

# Export all model information
all_models_info = manager.export_model_info()

# Delete a model
manager.delete_model("old_model_v1", model_type="baseline")
```

### Direct Model Save/Load

You can also save/load models directly without using ModelManager:

#### BaselineForecaster
```python
from backend.src.forecasting import BaselineForecaster

forecaster = BaselineForecaster()
forecaster.fit(df)

# Save directly
forecaster.save_model("my_forecast.joblib")

# Load directly
new_forecaster = BaselineForecaster()
new_forecaster.load_model("my_forecast.joblib")
```

#### RandomForestForecaster
```python
from backend.src.forecasting.models import RandomForestForecaster

model = RandomForestForecaster()
model.train(X, y)

# Save directly
model.save_model("my_rf_model.joblib")

# Load directly
new_model = RandomForestForecaster()
new_model.load_model("my_rf_model.joblib")
```

## Model Registry

The ModelManager maintains a central registry in `backend/models/model_registry.json`:

```json
{
  "baseline/demand_forecast_v1": {
    "type": "baseline",
    "path": "/path/to/models/baseline/demand_forecast_v1/demand_forecast_v1_model.joblib",
    "metadata_path": "/path/to/models/baseline/demand_forecast_v1/metadata.json",
    "saved_at": "2024-05-15T10:30:00.000000"
  },
  "random_forest/rf_demand_v1": {
    "type": "random_forest",
    "path": "/path/to/models/random_forest/rf_demand_v1/rf_demand_v1_model.joblib",
    "metadata_path": "/path/to/models/random_forest/rf_demand_v1/metadata.json",
    "saved_at": "2024-05-15T11:45:00.000000"
  }
}
```

## Metadata Files

Each saved model includes a `metadata.json` file with:

### BaselineForecaster Metadata
```json
{
  "model_type": "BaselineForecaster",
  "model_name": "demand_forecast_v1",
  "saved_at": "2024-05-15T10:30:00.000000",
  "model_info": {
    "model_type": "BaselineForecaster",
    "window_size": 7,
    "alpha": 0.3,
    "is_trained": true,
    "scaler_mean": 1234.56,
    "scaler_std": 234.12
  },
  "custom_metadata": {
    "description": "Monthly demand forecast",
    "data_source": "sales_data.csv"
  }
}
```

### RandomForestForecaster Metadata
```json
{
  "model_type": "RandomForestForecaster",
  "model_name": "rf_demand_v1",
  "saved_at": "2024-05-15T11:45:00.000000",
  "model_info": {
    "model_type": "RandomForest",
    "n_estimators": 100,
    "max_depth": 15,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "random_state": 42,
    "feature_count": 25,
    "is_trained": true,
    "has_scaler": true,
    "training_metrics": {...}
  },
  "training_metrics": {...},
  "custom_metadata": {...}
}
```

## Features

- **Model Persistence**: Save and load trained models using joblib
- **Metadata Tracking**: Store model configuration, training metrics, and custom metadata
- **Versioning**: Support multiple versions of models with clear naming
- **Central Registry**: Maintain a registry of all saved models
- **Organized Storage**: Models organized by type in separate directories
- **Easy Management**: List, delete, and export model information
- **Error Handling**: Comprehensive error handling and logging

## Dependencies

- **joblib**: For model serialization (already in requirements.txt)
- **scikit-learn**: For models (already in requirements.txt)

## Best Practices

1. **Use ModelManager**: For centralized model management, always use `ModelManager` or `get_model_manager()`
2. **Meaningful Names**: Use descriptive model names (e.g., "demand_forecast_v1", "revenue_rf_monthly")
3. **Metadata**: Always include relevant metadata when saving models
4. **Versioning**: Use version numbers in model names for tracking different iterations
5. **Error Handling**: Handle `FileNotFoundError` and `ValueError` when loading models
6. **Cleanup**: Delete old models to manage storage space

## Example: Complete Workflow

```python
from backend.src.forecasting import BaselineForecaster, get_model_manager
import pandas as pd

# Load data
df = pd.read_csv("sales_data.csv")

# Create and train model
forecaster = BaselineForecaster(window_size=14, alpha=0.2)
fit_stats = forecaster.fit(df)

# Evaluate
metrics = forecaster.evaluate(df)
print(f"Model metrics: {metrics}")

# Save with metadata
manager = get_model_manager()
model_path = manager.save_baseline_model(
    model=forecaster,
    model_name="demand_v2_production",
    metadata={
        "environment": "production",
        "window_size": 14,
        "alpha": 0.2,
        "metrics": metrics,
        "trained_on": "sales_data.csv"
    }
)

# Later... load and use
loaded_model = manager.load_baseline_model("demand_v2_production")
future_forecast = loaded_model.predict(df, periods=30)

# Get information about saved models
all_models = manager.list_models()
model_info = manager.export_model_info()
```
