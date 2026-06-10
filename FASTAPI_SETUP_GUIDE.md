# FastAPI Setup Guide

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Copy `.env.example` to `.env` and update the configuration:

```bash
cp .env.example .env
```

### 3. Run the Server

```bash
cd backend
python -m api.main
```

The API will start at `http://localhost:8000`

### 4. Access Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## 📁 Project Structure

```
backend/
├── api/
│   ├── __init__.py          # Router exports
│   ├── main.py              # FastAPI app initialization
│   ├── routes.py            # API route definitions
├── src/
│   ├── forecasting/         # ML forecasting models
│   ├── rag/                 # RAG pipeline
│   └── utils/               # Utility functions
├── data/
│   ├── sales_data.csv       # Training data
│   └── documents/           # Documents for RAG
├── models/                  # Pre-trained models
├── app.py                   # Legacy Flask app (deprecated)
├── config.py                # Configuration
└── requirements.txt         # Python dependencies
```

---

## 🔌 API Endpoints

### Health & Status
- `GET /health` - Health check
- `GET /api/status` - API status
- `GET /` - Root endpoint with available endpoints

### Forecasting
- `POST /api/forecast` - Create a new forecast
- `GET /api/forecast/{forecast_id}` - Get forecast results

### RAG Pipeline
- `POST /api/rag/query` - Query with RAG pipeline
- `POST /api/rag/batch-query` - Batch queries

---

## 🛠️ Development Commands

### Run in Development Mode (with hot reload)
```bash
cd backend
python -m api.main
```

### Run with Uvicorn Directly
```bash
cd backend
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### Run Tests
```bash
cd backend
pytest
```

### Format Code
```bash
black backend/
```

### Lint Code
```bash
flake8 backend/
```

---

## 🔑 Key Features

✅ **FastAPI Framework** - Modern, fast Python framework
✅ **Auto-generated Docs** - Swagger UI and ReDoc
✅ **CORS Support** - Cross-origin requests handling
✅ **Error Handling** - Global exception handlers
✅ **Environment Config** - .env support via python-dotenv
✅ **Async Support** - Full async/await support
✅ **Type Hints** - Full type checking with Pydantic
✅ **Logging** - Structured logging
✅ **Modular Routes** - Organized route structure

---

## 📊 Integrations

### RAG Pipeline Integration
Uncomment the RAG pipeline routes in `main.py` to enable:
- Document retrieval
- LLM-powered question answering
- Batch query processing

### ML Forecasting Integration
Uncomment the forecasting routes in `main.py` to enable:
- Sales forecasting
- Model inference
- Result retrieval

---

## ⚙️ Configuration

### Environment Variables
See `.env.example` for all available configuration options:

- `API_HOST` - Server host (default: 0.0.0.0)
- `API_PORT` - Server port (default: 8000)
- `DEBUG` - Debug mode (default: False)
- `CORS_ORIGINS` - CORS allowed origins (default: *)
- `DATABASE_URL` - Database connection string
- `REDIS_URL` - Redis connection string
- `OPENAI_API_KEY` - OpenAI API key (for LLM)

---

## 🐛 Troubleshooting

### Port Already in Use
```bash
# Use a different port
uvicorn api.main:app --port 8001
```

### Module Import Errors
```bash
# Make sure you're in the backend directory
cd backend

# Add backend to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Dependencies Missing
```bash
# Reinstall all dependencies
pip install --upgrade -r requirements.txt
```

---

## 📚 Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Uvicorn Documentation](https://www.uvicorn.org/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [Python-dotenv](https://github.com/theskumar/python-dotenv)

---

## 📝 Next Steps

1. **Implement Forecasting Endpoints**
   - Connect ML models from `src/forecasting/`
   - Add request validation with Pydantic models
   - Implement result caching

2. **Integrate RAG Pipeline**
   - Enable RAG endpoints in `main.py`
   - Add document management endpoints
   - Configure LLM providers

3. **Add Authentication**
   - JWT token support
   - API key authentication
   - Role-based access control

4. **Production Deployment**
   - Use Gunicorn with Uvicorn workers
   - Add database migrations
   - Set up monitoring and logging
   - Configure SSL/TLS

---

## ✅ Setup Checklist

- [ ] Python 3.8+ installed
- [ ] Dependencies installed: `pip install -r requirements.txt`
- [ ] `.env` file configured
- [ ] Server starts successfully: `python -m api.main`
- [ ] API docs accessible: http://localhost:8000/docs
- [ ] Health check passes: http://localhost:8000/health
