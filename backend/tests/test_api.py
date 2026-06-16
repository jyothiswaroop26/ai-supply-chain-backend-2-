"""
Tests for the FastAPI Supply Chain Backend.

Covers:
- GET  /           root info endpoint
- GET  /health     health check
- GET  /status     (via router, also at /api/status)
- POST /ask        RAG pipeline endpoint
- POST /forecast   demand forecasting endpoint
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

# Make sure the backend package is importable when running from any cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.main import app  # noqa: E402  (import after sys.path fix)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    """Module-scoped TestClient so the lifespan runs once per test module."""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def mock_rag_pipeline():
    """Return a mock RAG pipeline whose .query() returns a predictable dict."""
    pipeline = MagicMock()
    pipeline.query.return_value = {
        "query": "test",
        "answer": "This is a test answer.",
        "retrieved_documents": [
            {"source": "inventory_notes.txt", "content": "Sample content", "score": 0.9}
        ],
    }
    return pipeline


@pytest.fixture
def sample_sales_df():
    """Minimal sales DataFrame that BaselineForecaster can consume."""
    return pd.DataFrame(
        {
            "order_date": pd.date_range("2024-01-01", periods=30, freq="D").astype(str),
            "quantity": np.random.randint(1, 50, size=30).astype(float),
            "unit_price": [10.0] * 30,
            "discount": [0.0] * 30,
        }
    )


# ---------------------------------------------------------------------------
# Root endpoint
# ---------------------------------------------------------------------------

class TestRootEndpoint:
    def test_returns_200(self, client):
        response = client.get("/")
        assert response.status_code == 200

    def test_response_has_required_keys(self, client):
        data = client.get("/").json()
        assert "message" in data
        assert "version" in data
        assert "endpoints" in data

    def test_endpoints_map_contains_ask_and_forecast(self, client):
        endpoints = client.get("/").json()["endpoints"]
        assert "ask" in endpoints
        assert "health" in endpoints


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_status_is_healthy(self, client):
        data = client.get("/health").json()
        assert data["status"] == "healthy"

    def test_response_schema(self, client):
        data = client.get("/health").json()
        assert "service" in data
        assert "version" in data


# ---------------------------------------------------------------------------
# Status endpoint  (/status  and  /api/status)
# ---------------------------------------------------------------------------

class TestStatusEndpoint:
    def test_unprefixed_returns_200(self, client):
        response = client.get("/status")
        assert response.status_code == 200

    def test_prefixed_returns_200(self, client):
        response = client.get("/api/status")
        assert response.status_code == 200

    def test_message_field(self, client):
        data = client.get("/status").json()
        assert data.get("message") == "API is running"


# ---------------------------------------------------------------------------
# /ask  endpoint
# ---------------------------------------------------------------------------

class TestAskEndpoint:
    # --- validation errors (no mock needed) ---

    def test_empty_query_returns_422(self, client):
        response = client.post("/ask", json={"query": ""})
        assert response.status_code == 422

    def test_missing_query_returns_422(self, client):
        response = client.post("/ask", json={"num_retrieval": 5})
        assert response.status_code == 422

    def test_num_retrieval_too_high_returns_422(self, client):
        response = client.post(
            "/ask", json={"query": "test", "num_retrieval": 25}
        )
        assert response.status_code == 422

    def test_temperature_out_of_range_returns_422(self, client):
        response = client.post(
            "/ask", json={"query": "test", "temperature": 2.0}
        )
        assert response.status_code == 422

    def test_max_tokens_below_minimum_returns_422(self, client):
        response = client.post(
            "/ask", json={"query": "test", "max_tokens": 10}
        )
        assert response.status_code == 422

    # --- successful path (mock pipeline) ---

    def test_valid_query_returns_200(self, client, mock_rag_pipeline):
        with patch("api.main._get_or_create_rag_pipeline", return_value=mock_rag_pipeline):
            response = client.post(
                "/ask",
                json={
                    "query": "What is the inventory status?",
                    "num_retrieval": 3,
                    "max_tokens": 128,
                    "temperature": 0.5,
                },
            )
        assert response.status_code == 200

    def test_valid_query_response_contains_answer(self, client, mock_rag_pipeline):
        with patch("api.main._get_or_create_rag_pipeline", return_value=mock_rag_pipeline):
            data = client.post(
                "/ask", json={"query": "What are the shipping delays?"}
            ).json()
        assert "answer" in data

    def test_pipeline_query_called_with_correct_params(self, client, mock_rag_pipeline):
        payload = {
            "query": "supplier list",
            "num_retrieval": 4,
            "max_tokens": 200,
            "temperature": 0.6,
        }
        with patch("api.main._get_or_create_rag_pipeline", return_value=mock_rag_pipeline):
            client.post("/ask", json=payload)
        mock_rag_pipeline.query.assert_called_once_with(
            query="supplier list",
            num_retrieval=4,
            max_tokens=200,
            temperature=0.6,
        )

    def test_pipeline_value_error_returns_400(self, client):
        bad_pipeline = MagicMock()
        bad_pipeline.query.side_effect = ValueError("pipeline not ready")
        with patch("api.main._get_or_create_rag_pipeline", return_value=bad_pipeline):
            response = client.post("/ask", json={"query": "test"})
        assert response.status_code == 400

    def test_pipeline_internal_error_returns_500(self, client):
        bad_pipeline = MagicMock()
        bad_pipeline.query.side_effect = RuntimeError("unexpected failure")
        with patch("api.main._get_or_create_rag_pipeline", return_value=bad_pipeline):
            response = client.post("/ask", json={"query": "test"})
        assert response.status_code == 500

    @pytest.mark.parametrize(
        "query",
        [
            "What is the current inventory status?",
            "Show me the sales forecast",
            "What are the shipping delays?",
            "List all suppliers",
            "What is the current demand?",
        ],
    )
    def test_various_queries_accepted(self, client, mock_rag_pipeline, query):
        with patch("api.main._get_or_create_rag_pipeline", return_value=mock_rag_pipeline):
            response = client.post("/ask", json={"query": query})
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# /forecast  endpoint
# ---------------------------------------------------------------------------

class TestForecastEndpoint:
    # --- validation errors ---

    def test_invalid_freq_returns_422(self, client):
        response = client.post("/forecast", json={"freq": "X"})
        assert response.status_code == 422

    def test_periods_zero_returns_422(self, client):
        response = client.post("/forecast", json={"periods": 0})
        assert response.status_code == 422

    def test_periods_too_large_returns_422(self, client):
        response = client.post("/forecast", json={"periods": 400})
        assert response.status_code == 422

    # --- successful path (mock sales data) ---

    def test_default_payload_returns_200(self, client, sample_sales_df):
        with patch("api.main._get_sales_data", return_value=sample_sales_df):
            response = client.post("/forecast", json={})
        assert response.status_code == 200

    def test_response_schema(self, client, sample_sales_df):
        with patch("api.main._get_sales_data", return_value=sample_sales_df):
            data = client.post("/forecast", json={"periods": 7, "freq": "D"}).json()
        assert "periods" in data
        assert "freq" in data
        assert "value_col" in data
        assert "forecasts" in data

    def test_forecasts_contain_expected_methods(self, client, sample_sales_df):
        with patch("api.main._get_sales_data", return_value=sample_sales_df):
            forecasts = client.post(
                "/forecast", json={"periods": 7, "methods": ["sma", "trend"]}
            ).json()["forecasts"]
        assert "sma" in forecasts
        assert "trend" in forecasts

    def test_forecast_length_matches_periods(self, client, sample_sales_df):
        periods = 5
        with patch("api.main._get_sales_data", return_value=sample_sales_df):
            forecasts = client.post(
                "/forecast", json={"periods": periods}
            ).json()["forecasts"]
        for method, values in forecasts.items():
            assert len(values) == periods, f"Method '{method}' returned wrong length"

    def test_forecast_values_are_numeric(self, client, sample_sales_df):
        with patch("api.main._get_sales_data", return_value=sample_sales_df):
            forecasts = client.post("/forecast", json={"periods": 3}).json()["forecasts"]
        for method, values in forecasts.items():
            for v in values:
                assert isinstance(v, (int, float)), f"Non-numeric value in {method}: {v}"

    def test_weekly_freq(self, client, sample_sales_df):
        with patch("api.main._get_sales_data", return_value=sample_sales_df):
            response = client.post("/forecast", json={"periods": 4, "freq": "W"})
        assert response.status_code == 200

    def test_missing_sales_data_returns_404(self, client):
        with patch("api.main._get_sales_data", side_effect=FileNotFoundError("no file")):
            response = client.post("/forecast", json={})
        assert response.status_code == 404

    def test_internal_error_returns_500(self, client):
        with patch("api.main._get_sales_data", side_effect=RuntimeError("db down")):
            response = client.post("/forecast", json={})
        assert response.status_code == 500


# ---------------------------------------------------------------------------
# Content-type / general API behaviour
# ---------------------------------------------------------------------------

class TestAPIBehaviour:
    def test_all_responses_are_json(self, client):
        for path in ("/", "/health", "/status"):
            ct = client.get(path).headers["content-type"]
            assert "application/json" in ct, f"{path} did not return JSON"

    def test_unknown_route_returns_404(self, client):
        response = client.get("/this-does-not-exist")
        assert response.status_code == 404

    def test_openapi_schema_available(self, client):
        data = client.get("/openapi.json").json()
        assert "openapi" in data
        assert "paths" in data

    def test_swagger_ui_available(self, client):
        response = client.get("/docs")
        assert response.status_code == 200

    def test_redoc_ui_available(self, client):
        response = client.get("/redoc")
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    raise SystemExit(
        pytest.main([__file__, "-v", "--tb=short", "-s", "--color=yes"])
    )
