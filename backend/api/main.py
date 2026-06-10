"""
FastAPI Main Application
Entry point for the AI Supply Chain Backend API
"""
import os
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from src.rag.rag_pipeline import create_rag_pipeline

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


def _get_documents_dir() -> str:
    """Resolve documents directory path relative to backend folder."""
    return str(Path(__file__).resolve().parent.parent / "data" / "documents")


def _get_or_create_rag_pipeline():
    """Create and cache RAG pipeline on first use."""
    pipeline = app_state.get("rag_pipeline")
    if pipeline is not None:
        return pipeline

    pipeline = create_rag_pipeline(llm_provider_type="local")
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

# API status endpoint
@app.get("/api/status", tags=["Status"])
async def api_status():
    """
    API status endpoint
    """
    return {
        "message": "API is running",
        "initialized": app_state.get('initialized', False)
    }

# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint providing API information
    """
    return {
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
    except Exception as exc:
        logger.error("Failed to process /ask request: %s", str(exc), exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to process query") from exc

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

# RAG Pipeline routes (example - uncomment when RAG pipeline is ready)
# from src.rag.rag_pipeline import create_rag_pipeline
# @app.post("/api/rag/query", tags=["RAG"])
# async def rag_query(query: str, num_retrieval: int = 5):
#     """Query the RAG pipeline"""
#     try:
#         pipeline = create_rag_pipeline()
#         result = pipeline.query(query, num_retrieval=num_retrieval)
#         return result
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# Forecasting endpoints (example - ready for implementation)
# @app.post("/api/forecast", tags=["Forecasting"])
# async def create_forecast(data: dict):
#     """Create a sales forecast using the model"""
#     pass

# @app.get("/api/forecast/{forecast_id}", tags=["Forecasting"])
# async def get_forecast(forecast_id: str):
#     """Retrieve a forecast result"""
#     pass

if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv('API_HOST', '0.0.0.0')
    port = int(os.getenv('API_PORT', 8000))
    reload = os.getenv('DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting server at {host}:{port}")
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )
