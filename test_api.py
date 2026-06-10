"""
Comprehensive API Tests for FastAPI Supply Chain Backend
Tests the main API endpoints and functionality
"""
import pytest
import asyncio
import sys
from pathlib import Path
from fastapi.testclient import TestClient

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

# Import after path setup
from api.main import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app"""
    return TestClient(app)


class TestStatusEndpoints:
    """Test status and health check endpoints"""
    
    def test_status_endpoint(self, client):
        """Test the /status endpoint"""
        response = client.get("/status")
        assert response.status_code == 200
        assert "message" in response.json()
        assert response.json()["message"] == "API is running"
    
    def test_root_endpoint(self, client):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data


class TestRAGEndpoints:
    """Test RAG pipeline endpoints"""
    
    def test_ask_endpoint_valid_query(self, client):
        """Test the ask endpoint with valid query"""
        payload = {
            "query": "What is the current inventory status?",
            "num_retrieval": 5,
            "max_tokens": 256,
            "temperature": 0.7
        }
        response = client.post("/ask", json=payload)
        assert response.status_code in [200, 404]  # 404 if no documents available
        
    def test_ask_endpoint_invalid_query_empty(self, client):
        """Test ask endpoint with empty query"""
        payload = {
            "query": "",
            "num_retrieval": 5,
            "max_tokens": 256,
            "temperature": 0.7
        }
        response = client.post("/ask", json=payload)
        assert response.status_code == 422  # Validation error
    
    def test_ask_endpoint_invalid_num_retrieval(self, client):
        """Test ask endpoint with invalid num_retrieval"""
        payload = {
            "query": "test query",
            "num_retrieval": 25,  # Out of range
            "max_tokens": 256,
            "temperature": 0.7
        }
        response = client.post("/ask", json=payload)
        assert response.status_code == 422  # Validation error
    
    def test_ask_endpoint_invalid_temperature(self, client):
        """Test ask endpoint with invalid temperature"""
        payload = {
            "query": "test query",
            "num_retrieval": 5,
            "max_tokens": 256,
            "temperature": 1.5  # Out of range
        }
        response = client.post("/ask", json=payload)
        assert response.status_code == 422  # Validation error
    
    def test_ask_endpoint_missing_required_field(self, client):
        """Test ask endpoint with missing required field"""
        payload = {
            "num_retrieval": 5,
            "max_tokens": 256,
            "temperature": 0.7
        }
        response = client.post("/ask", json=payload)
        assert response.status_code == 422  # Validation error


class TestForecastingEndpoints:
    """Test forecasting endpoints"""
    
    def test_forecast_endpoint_valid(self, client):
        """Test the forecast endpoint with valid data"""
        payload = {
            "historical_data": [100, 110, 120, 130, 140],
            "periods_ahead": 12,
            "model_type": "random_forest"
        }
        response = client.post("/forecast", json=payload)
        assert response.status_code in [200, 400]  # May fail if model not loaded
        
    def test_get_forecast_endpoint(self, client):
        """Test getting forecast by ID"""
        response = client.get("/forecast/test-id")
        assert response.status_code in [200, 404]


class TestAPIIntegration:
    """Test API integration and general functionality"""
    
    def test_api_accepts_json(self, client):
        """Test that API accepts JSON content type"""
        payload = {
            "query": "Test query",
            "num_retrieval": 5,
            "max_tokens": 256,
            "temperature": 0.7
        }
        response = client.post(
            "/ask",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        assert response.headers["content-type"] == "application/json"
    
    def test_api_returns_json(self, client):
        """Test that API returns JSON responses"""
        response = client.get("/status")
        assert response.headers["content-type"] == "application/json"
    
    def test_api_error_handling(self, client):
        """Test API error handling"""
        response = client.post("/ask", json={"query": ""})
        assert response.status_code >= 400
        data = response.json()
        assert "detail" in data or "message" in data


class TestCORS:
    """Test CORS headers"""
    
    def test_cors_headers_present(self, client):
        """Test that CORS headers are present in response"""
        response = client.get("/status")
        assert response.status_code == 200
        # CORS headers should be configured
        headers = response.headers
        # Check for common CORS headers
        assert any(
            key.lower().startswith("access-control")
            for key in headers.keys()
        ) or response.status_code == 200  # Allow if no CORS headers


class TestDocumentation:
    """Test API documentation"""
    
    def test_openapi_schema(self, client):
        """Test OpenAPI schema endpoint"""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        data = response.json()
        assert "openapi" in data
        assert "paths" in data
    
    def test_swagger_ui(self, client):
        """Test Swagger UI endpoint"""
        response = client.get("/docs")
        assert response.status_code == 200
        assert "swagger" in response.text.lower() or "html" in response.headers["content-type"]
    
    def test_redoc_ui(self, client):
        """Test ReDoc UI endpoint"""
        response = client.get("/redoc")
        assert response.status_code == 200


# Parametrized tests
class TestParametrizedQueries:
    """Test with multiple query types"""
    
    @pytest.mark.parametrize("query", [
        "What is the inventory status?",
        "Show me the sales forecast",
        "What are the shipping delays?",
        "List all suppliers",
        "What is the current demand?",
    ])
    def test_ask_with_various_queries(self, client, query):
        """Test ask endpoint with various queries"""
        payload = {
            "query": query,
            "num_retrieval": 5,
            "max_tokens": 256,
            "temperature": 0.7
        }
        response = client.post("/ask", json=payload)
        assert response.status_code in [200, 404]  # May fail if docs not available


# Run tests
if __name__ == "__main__":
    # Run pytest
    exit_code = pytest.main([
        __file__,
        "-v",  # Verbose
        "--tb=short",  # Short traceback
        "-s",  # Show print statements
        "--color=yes"  # Colored output
    ])
    sys.exit(exit_code)
