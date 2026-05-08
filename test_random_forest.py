"""Test script to verify Random Forest model functionality."""

if __name__ == "__main__":
    try:
        # Test imports
        import pandas as pd
        import numpy as np
        from sklearn.ensemble import RandomForestRegressor
        print("✓ All dependencies imported successfully")
        
        # Test model import
        from backend.src.forecasting.models.random_forest import RandomForestForecaster
        print("✓ RandomForestForecaster imported successfully")
        
        # Test model initialization
        rf = RandomForestForecaster()
        print("✓ Model initialized successfully")
        
        # Test model info
        info = rf.get_model_info()
        print(f"✓ Model Type: {info['model_type']}")
        print(f"✓ Estimators: {info['n_estimators']}")
        print(f"✓ Max Depth: {info['max_depth']}")
        
        # Test with sample data
        sample_X = pd.DataFrame(
            np.random.randn(100, 5),
            columns=[f'feature_{i}' for i in range(5)]
        )
        sample_y = pd.Series(np.random.randn(100))
        
        metrics = rf.train(sample_X, sample_y, test_size=0.2, verbose=False)
        print(f"✓ Model trained successfully")
        print(f"  - Train MAE: {metrics['train_mae']:.4f}")
        print(f"  - Test MAE: {metrics['test_mae']:.4f}")
        print(f"  - Test R²: {metrics['test_r2']:.4f}")
        
        # Test predictions
        pred = rf.predict(sample_X.head(10))
        print(f"✓ Predictions generated: {len(pred)} samples")
        
        # Test feature importance
        importance = rf.get_feature_importance(top_n=3)
        print(f"✓ Top features: {importance['feature'].tolist()}")
        
        print("\n" + "="*50)
        print("✓ ALL TESTS PASSED - Random Forest Model is Working!")
        print("="*50)
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)
