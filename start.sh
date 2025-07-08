#!/bin/bash

# Chat with PDF - Multi-Agent System Startup Script
set -e

echo "🚀 Starting Chat with PDF Multi-Agent System..."

# Default values
DEFAULT_HOST="0.0.0.0"
DEFAULT_PORT="8000"
VENV_DIR=".venv"

# Load environment variables if .env exists
if [ -f ".env" ]; then
    echo "📄 Loading environment variables from .env file..."
    export $(grep -v '^#' .env | xargs)
else
    echo "⚠️  No .env file found. Please copy env.example to .env and configure your API keys."
    echo "   cp env.example .env"
    exit 1
fi

# Check for required API keys
if [ -z "$GOOGLE_API_KEY" ]; then
    echo "❌ GOOGLE_API_KEY is not set in .env file"
    exit 1
fi

if [ -z "$TAVILY_API_KEY" ]; then
    echo "❌ TAVILY_API_KEY is not set in .env file"
    exit 1
fi

# Use environment variables or defaults
HOST=${HOST:-$DEFAULT_HOST}
PORT=${PORT:-$DEFAULT_PORT}

echo "📦 Setting up Python virtual environment..."

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "   Creating new virtual environment..."
    python3 -m venv $VENV_DIR
else
    echo "   Using existing virtual environment..."
fi

# Activate virtual environment
echo "   Activating virtual environment..."
source $VENV_DIR/bin/activate

# Upgrade pip
echo "   Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📚 Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating data directories..."
mkdir -p data/pdfs
mkdir -p data/vectorstore
mkdir -p logs

# Check if PDFs exist for ingestion
PDF_COUNT=$(find data/pdfs -name "*.pdf" 2>/dev/null | wc -l)
if [ $PDF_COUNT -gt 0 ]; then
    echo "📄 Found $PDF_COUNT PDF file(s) in data/pdfs/"
    echo "   You can ingest them later with: python scripts/ingest_pdfs.py default"
else
    echo "📄 No PDF files found in data/pdfs/ directory"
    echo "   Add PDF files to data/pdfs/ and run: python scripts/ingest_pdfs.py default"
fi

echo ""
echo "🎯 Configuration:"
echo "   Host: $HOST"
echo "   Port: $PORT"
echo "   Log Level: ${LOG_LEVEL:-INFO}"
echo "   Model: ${MODEL_NAME:-gemini-pro}"
echo "   Embedding Model: ${EMBEDDING_MODEL:-models/text-embedding-004}"
echo ""

echo "🌟 Starting FastAPI application..."
echo "   API Documentation: http://$HOST:$PORT/docs"
echo "   Health Check: http://$HOST:$PORT/health"
echo ""

# Start the application
python -m uvicorn src.api.main:app \
    --host $HOST \
    --port $PORT \
    --reload \
    --log-level ${LOG_LEVEL:-info} 