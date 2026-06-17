"""
Full system pipeline tests for the AI Supply Chain backend.

Validates the practical chain:
Data -> Model -> API -> UI-facing response contract
"""

from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

# Ensure backend imports work regardless of where pytest is executed from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.main import app  # noqa: E402
from src.forecasting.model import BaselineForecaster  # noqa: E402


@pytest.fixture(scope="module")
def client():
    """Shared API client for integration-style tests."""
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def sales_df():
    """Load real sales data used by the forecasting endpoint."""
    csv_path = Path(__file__).resolve().parent.parent / "data" / "sales_data.csv"
    assert csv_path.exists(), f"Expected sales data file at {csv_path}"

    df = pd.read_csv(csv_path)
    assert not df.empty
    assert "order_date" in df.columns
    assert "quantity" in df.columns

    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df = df.dropna(subset=["quantity"])
    return df


class TestFullSystemPipeline:
    """End-to-end pipeline checks across layers."""

    def test_data_to_model_generates_forecast(self, sales_df):
        """Validate raw data can be consumed by the forecasting model."""
        forecaster = BaselineForecaster()
        result = forecaster.predict(
            sales_df,
            periods=7,
            date_col="order_date",
            value_col="quantity",
            freq="D",
            methods=["sma", "trend"],
        )

        assert "sma" in result
        assert "trend" in result
        assert "ensemble" in result
        assert len(result["sma"]) == 7
        assert len(result["trend"]) == 7
        assert np.isfinite(result["sma"]).all()
        assert np.isfinite(result["trend"]).all()

    def test_model_to_api_forecast_consistency(self, client, sales_df):
        """Validate API forecast values match direct model output on same data."""
        periods = 5
        methods = ["sma", "trend"]

        forecaster = BaselineForecaster()
        model_result = forecaster.predict(
            sales_df,
            periods=periods,
            date_col="order_date",
            value_col="quantity",
            freq="D",
            methods=methods,
        )

        response = client.post(
            "/forecast",
            json={
                "periods": periods,
                "freq": "D",
                "methods": methods,
                "date_col": "order_date",
                "value_col": "quantity",
            },
        )
        assert response.status_code == 200

        api_result = response.json()
        api_forecasts = api_result["forecasts"]

        for method in ("sma", "trend", "ensemble"):
            assert method in api_forecasts
            assert len(api_forecasts[method]) == periods
            assert api_forecasts[method] == pytest.approx(model_result[method].tolist(), rel=1e-6, abs=1e-6)

    def test_api_to_ui_forecast_contract(self, client):
        """Validate forecast payload is chart-ready for UI consumption."""
        response = client.post(
            "/forecast",
            json={"periods": 6, "freq": "W", "methods": ["sma", "trend"]},
        )
        assert response.status_code == 200

        data = response.json()
        assert data["periods"] == 6
        assert data["freq"] == "W"
        assert data["value_col"] == "quantity"
        assert "forecasts" in data

        for series_name, values in data["forecasts"].items():
            assert isinstance(series_name, str)
            assert isinstance(values, list)
            assert len(values) == 6
            assert all(isinstance(v, (int, float)) for v in values)

    def test_rag_api_to_ui_contract(self, client):
        """Validate /ask returns a stable UI-facing schema."""
        mock_pipeline = MagicMock()
        mock_pipeline.query.return_value = {
            "query": "What is current inventory?",
            "answer": "Inventory is stable this week.",
            "num_retrieved": 1,
            "retrieved_documents": [
                {
                    "id": "doc_1",
                    "rank": 1,
                    "source": "inventory_notes.txt",
                    "score": 0.92,
                    "content": "Inventory levels are stable across key SKUs.",
                    "metadata": {"department": "warehouse"},
                }
            ],
        }

        with patch("api.main._get_or_create_rag_pipeline", return_value=mock_pipeline):
            response = client.post(
                "/ask",
                json={
                    "query": "What is current inventory?",
                    "num_retrieval": 4,
                    "max_tokens": 128,
                    "temperature": 0.4,
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "What is current inventory?"
        assert isinstance(data.get("answer"), str)
        assert isinstance(data.get("retrieved_documents"), list)
        assert len(data["retrieved_documents"]) == 1

        first_doc = data["retrieved_documents"][0]
        assert "source" in first_doc
        assert "content" in first_doc
        assert "score" in first_doc

    def test_openapi_exposes_ui_required_endpoints(self, client):
        """Validate docs schema exposes endpoints the frontend can bind to."""
        response = client.get("/openapi.json")
        assert response.status_code == 200

        schema = response.json()
        paths = schema.get("paths", {})
        assert "/forecast" in paths
        assert "/ask" in paths
        assert "/status" in paths or "/api/status" in paths
