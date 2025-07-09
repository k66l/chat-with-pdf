"""FastAPI application for Chat with PDF multi-agent system."""

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from ..core.config import settings, ensure_directories
from .ask import router as ask_router
from .health import router as health_router
from .memory import router as memory_router
from .middleware import RequestResponseLoggingMiddleware, RequestMetricsMiddleware, global_metrics

# Setup logging
import logging
import sys
from structlog.dev import ConsoleRenderer

# Configure standard library logging
logging.basicConfig(
    format="%(message)s",
    stream=sys.stdout,
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
)

# Configure structlog for readable console output
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="ISO"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        # Use ConsoleRenderer for readable output in development
        ConsoleRenderer(colors=True)
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Chat with PDF - Multi-Agent System",
    description="A multi-agent system for intelligent question-answering over PDF documents",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add middleware (order matters - first added is outermost)
# Add CORS middleware first
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add request/response logging middleware
app.add_middleware(RequestResponseLoggingMiddleware)

# Add metrics collection middleware
app.add_middleware(RequestMetricsMiddleware)

# Include routers
app.include_router(ask_router, tags=["Questions"])
app.include_router(health_router, tags=["Health"])
app.include_router(memory_router, tags=["Memory"])


@app.on_event("startup")
async def startup_event():
    """Initialize the application."""
    try:
        logger.info("Starting Chat with PDF application")

        # Ensure directories exist
        ensure_directories()

        logger.info("Application startup completed")

    except Exception as e:
        logger.error("Error during startup", error=str(e))
        raise


# Add endpoint to view metrics
@app.get("/metrics", tags=["Monitoring"])
async def get_metrics():
    """Get application metrics and statistics."""
    return global_metrics.get_metrics()


if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
        log_level=settings.log_level.lower(),
        access_log=True,
        use_colors=True
    )
