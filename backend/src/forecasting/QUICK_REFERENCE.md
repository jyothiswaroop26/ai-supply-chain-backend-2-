# Quick Reference: Model Saving & Loading

## TL;DR - Fast Implementation Guide

### Install Dependencies
All dependencies are already in `requirements.txt`:
- joblib 1.3.1
- scikit-learn 1.3.0

### Save a Model

**Baseline Model:**
```python
from backend.src.forecasting import BaselineForecaster, get_model_manager

# Train
model = BaselineForecaster()
model.fit(df)

# Save
manager = get_model_manager()
manager.save_baseline_model(model, "my_baseline_v1", metadata={"version": "1.0"})
```

**Random Forest Model:**
```python
from backend.src.forecasting.models import RandomForestForecaster
from backend.src.forecasting import get_model_manager

# Train
model = RandomForestForecaster()
model.train(X, y)

# Save
manager = get_model_manager()
manager.save_random_forest_model(model, "my_rf_v1", metadata={"version": "1.0"})
```

### Load and Use a Model

**Baseline:**
```python
from backend.src.forecasting import get_model_manager

manager = get_model_manager()
model = manager.load_baseline_model("my_baseline_v1")
predictions = model.predict(df, periods=30)
```

**Random Forest:**
```python
from backend.src.forecasting import get_model_manager

manager = get_model_manager()
model = manager.load_random_forest_model("my_rf_v1")
predictions = model.predict(X)
```

### Manage Models

```python
from backend.src.forecasting import get_model_manager

manager = get_model_manager()

# List all models
models = manager.list_models()

# Get model details
metadata = manager.get_model_metadata("my_baseline_v1", "baseline")

# Delete model
manager.delete_model("old_model_v1", "baseline")

# Export all info
all_info = manager.export_model_info()
```

## Directory Structure

```
backend/models/
├── baseline/
│   └── my_baseline_v1/
│       ├── my_baseline_v1_model.joblib
│       └── metadata.json
├── random_forest/
│   └── my_rf_v1/
│       ├── my_rf_v1_model.joblib
│       └── metadata.json
└── model_registry.json
```

## ModelManager Methods

| Method | Purpose | Returns |
|--------|---------|---------|
| `save_baseline_model(model, name, metadata)` | Save baseline model | Model path (str) |
| `save_random_forest_model(model, name, metadata)` | Save RF model | Model path (str) |
| `load_baseline_model(name)` | Load baseline model | BaselineForecaster |
| `load_random_forest_model(name)` | Load RF model | RandomForestForecaster |
| `get_model_metadata(name, type)` | Get model info | Dict with metadata |
| `list_models()` | List all saved models | Dict by type |
| `delete_model(name, type)` | Delete model | None |
| `export_model_info()` | Export registry | Dict with all models |

## Error Handling

```python
from backend.src.forecasting import get_model_manager

manager = get_model_manager()

try:
    model = manager.load_baseline_model("nonexistent")
except ValueError as e:
    print(f"Model not found: {e}")
except FileNotFoundError as e:
    print(f"Model file missing: {e}")
```

## Files Modified/Created

### New Files
- `backend/src/forecasting/model_manager.py` - Manager utility
- `backend/src/forecasting/examples_model_saving.py` - Usage examples
- `backend/models/README.md` - Full documentation
- `backend/models/.gitkeep` - Directory marker
- `TASK_COMPLETION_SUMMARY.md` - Completion summary

### Modified Files
- `backend/src/forecasting/model.py` - Added save/load to BaselineForecaster
- `backend/src/forecasting/__init__.py` - Export ModelManager

### Created Directories
- `backend/models/` - Model storage root
- `backend/models/baseline/` - Baseline models
- `backend/models/random_forest/` - RF models

## Running Examples

```bash
cd backend
python -m src.forecasting.examples_model_saving
```

## Common Patterns

### Pattern 1: Train Once, Load Many Times
```python
# Script 1: Train and save
model = BaselineForecaster()
model.fit(df)
manager = get_model_manager()
manager.save_baseline_model(model, "prod_model")

# Script 2-N: Load and predict
from backend.src.forecasting import get_model_manager
manager = get_model_manager()
model = manager.load_baseline_model("prod_model")
results = model.predict(df, periods=7)
```

### Pattern 2: Model Versioning
```python
# v1
manager.save_baseline_model(model_v1, "demand_forecast_v1")

# v2
manager.save_baseline_model(model_v2, "demand_forecast_v2")

# Use latest
model = manager.load_baseline_model("demand_forecast_v2")
```

### Pattern 3: Model with Metrics
```python
model = RandomForestForecaster()
metrics = model.train(X, y)

manager.save_random_forest_model(
    model, 
    "production_model",
    metadata={
        "metrics": metrics,
        "date": datetime.now().isoformat(),
        "notes": "Tuned hyperparameters"
    }
)
```

## Troubleshooting

**Q: Model not found error**
- A: Ensure model name is correct. Use `manager.list_models()` to see available models

**Q: Where are models stored?**
- A: `backend/models/` directory (created automatically)

**Q: Can I save models to custom location?**
- A: Yes: `manager = ModelManager(models_dir="/custom/path")`

**Q: How do I update/retrain a model?**
- A: Save with the same name to overwrite, or use versioning (v1, v2, etc.)

**Q: Can I use both save methods (direct + manager)?**
- A: Yes, but manager provides better organization and metadata tracking

## Performance Notes

- **Save Time**: Typically < 1 second for most models
- **Load Time**: Typically < 1 second for most models
- **Storage**: Models are compressed with joblib, typically 1-50 MB depending on complexity
- **Metadata**: JSON files are minimal overhead

## Security Notes

- Models contain trained data patterns but not raw data
- Store sensitive models in secure directories
- Access control managed by filesystem permissions
- Registry file is plain JSON (humanreadable, not encrypted)

## Next Steps

1. Integrate into API endpoints
2. Add model validation on load
3. Implement automatic versioning
4. Add monitoring for model drift
5. Consider cloud storage integration
