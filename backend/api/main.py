"""
FastAPI Main Application
Entry point for the AI Supply Chain Backend API
"""
import os
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import pandas as pd
from src.rag.rag_pipeline import create_rag_pipeline
from src.forecasting.model import BaselineForecaster

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Store app state
app_state = {}


class AskRequest(BaseModel):
    """Request payload for the ask endpoint."""

    query: str = Field(..., min_length=1, description="User question for the RAG pipeline")
    num_retrieval: int = Field(default=5, ge=1, le=20)
    max_tokens: int = Field(default=256, ge=50, le=2000)
    temperature: float = Field(default=0.7, ge=0.0, le=1.0)


class ForecastRequest(BaseModel):
    """Request payload for the forecast endpoint."""

    periods: int = Field(default=7, ge=1, le=365, description="Number of periods to forecast ahead")
    freq: str = Field(default="D", pattern="^(D|W|M)$", description="Frequency: D=daily, W=weekly, M=monthly")
    methods: Optional[List[str]] = Field(
        default=None,
        description="Forecast methods to use: 'sma', 'es', 'trend'. Defaults to all."
    )
    date_col: str = Field(default="order_date", description="Date column in the dataset")
    value_col: str = Field(default="quantity", description="Value column to forecast")


def _get_sales_data() -> pd.DataFrame:
    """Load sales data CSV relative to the backend folder."""
    csv_path = Path(__file__).resolve().parent.parent / "data" / "sales_data.csv"
    df = pd.read_csv(csv_path)
    # Coerce non-numeric quantity values to NaN then drop them
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df = df.dropna(subset=["quantity"])
    return df


def _get_documents_dir() -> str:
    """Resolve documents directory path relative to backend folder."""
    return str(Path(__file__).resolve().parent.parent / "data" / "documents")


def _get_or_create_rag_pipeline():
    """Create and cache RAG pipeline on first use."""
    pipeline = app_state.get("rag_pipeline")
    if pipeline is not None:
        return pipeline

    # Default to TF-IDF embeddings to avoid heavy model downloads on first request.
    embedding_model = os.getenv("RAG_EMBEDDING_MODEL", "tfidf")
    pipeline = create_rag_pipeline(
        llm_provider_type="local",
        embedding_model=embedding_model,
    )
    pipeline.load_documents(_get_documents_dir())
    app_state["rag_pipeline"] = pipeline
    return pipeline

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage app lifecycle - startup and shutdown events
    """
    # Startup
    logger.info("Starting AI Supply Chain Backend API")
    app_state['initialized'] = True
    yield
    # Shutdown
    logger.info("Shutting down AI Supply Chain Backend API")
    app_state.clear()

# Initialize FastAPI app
app = FastAPI(
    title="AI Supply Chain Backend API",
    description="Backend API for AI-powered supply chain forecasting and RAG pipeline",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
allowed_origins = os.getenv('CORS_ORIGINS', '*').split(',')
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint to verify API is running
    """
    return {
        "status": "healthy",
        "service": "AI Supply Chain Backend",
        "version": "1.0.0"
    }

# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint providing API information
    """
    return {
        "message": "AI Supply Chain Backend API",
        "service": "AI Supply Chain Backend API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "ask": "/ask",
            "status": "/api/status",
            "docs": "/docs",
            "redoc": "/redoc"
        }
    }


@app.post("/ask", tags=["RAG"])
async def ask_question(payload: AskRequest):
    """Ask a question to the RAG pipeline and return an answer with retrieved context."""
    try:
        pipeline = _get_or_create_rag_pipeline()
        return pipeline.query(
            query=payload.query,
            num_retrieval=payload.num_retrieval,
            max_tokens=payload.max_tokens,
            temperature=payload.temperature,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (ImportError, ModuleNotFoundError, FileNotFoundError) as exc:
        logger.warning("RAG resources unavailable for /ask request: %s", str(exc))
        raise HTTPException(status_code=404, detail="RAG resources are unavailable") from exc
    except Exception as exc:
        logger.error("Failed to process /ask request: %s", str(exc), exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to process query") from exc

@app.post("/forecast", tags=["Forecasting"])
async def forecast(payload: ForecastRequest):
    """Generate demand forecasts from historical sales data using the baseline forecaster."""
    try:
        df = _get_sales_data()
        forecaster = BaselineForecaster()
        result = forecaster.predict(
            df,
            periods=payload.periods,
            date_col=payload.date_col,
            value_col=payload.value_col,
            freq=payload.freq,
            methods=payload.methods,
        )
        # Convert numpy arrays to plain lists for JSON serialisation
        serialisable = {method: values.tolist() for method, values in result.items()}
        return {
            "periods": payload.periods,
            "freq": payload.freq,
            "value_col": payload.value_col,
            "forecasts": serialisable,
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        logger.error("Sales data file not found: %s", str(exc))
        raise HTTPException(status_code=404, detail="Sales data not found") from exc
    except Exception as exc:
        logger.error("Failed to process /forecast request: %s", str(exc), exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to generate forecast") from exc


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """
    Handle global exceptions with proper logging and response format
    """
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc) if os.getenv('DEBUG', 'False').lower() == 'true' else None
        }
    )

# Include routers
from api.routes import router as api_router
app.include_router(api_router, prefix="/api")
# Backward-compatible, unprefixed routes used by existing tests and clients.
app.include_router(api_router)

if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv('API_HOST', '0.0.0.0')
    port = int(os.getenv('API_PORT', 8000))
    reload = os.getenv('DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting server at {host}:{port}")
    uvicorn.run(
        "api.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )
