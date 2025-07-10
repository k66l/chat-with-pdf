#!/bin/bash

# Complete Docker Setup Script for Chat with PDF
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Default values
IMAGE_NAME="chat-with-pdf"
CONTAINER_NAME="chat-with-pdf-app"
COMPOSE_FILE="docker-compose.yml"

print_status "Starting complete Docker setup for Chat with PDF..."

# Function to check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker and try again."
        exit 1
    fi
    print_success "Docker is running"
}

# Function to check environment setup
check_environment() {
    print_status "Checking environment setup..."
    
    # Check if .env file exists
    if [ ! -f ".env" ]; then
        print_warning "No .env file found. Creating from env.example..."
        if [ -f "env.example" ]; then
            cp env.example .env
            print_warning "Please edit .env file and add your API keys:"
            print_warning "  - GOOGLE_API_KEY"
            print_warning "  - TAVILY_API_KEY"
            echo ""
            print_warning "Then run this script again."
            exit 1
        else
            print_error "env.example file not found!"
            exit 1
        fi
    fi
    
    # Check for required API keys
    source .env
    if [ -z "$GOOGLE_API_KEY" ] || [ "$GOOGLE_API_KEY" = "your_google_api_key_here" ]; then
        print_error "GOOGLE_API_KEY is not set in .env file"
        exit 1
    fi
    
    if [ -z "$TAVILY_API_KEY" ] || [ "$TAVILY_API_KEY" = "your_tavily_api_key_here" ]; then
        print_error "TAVILY_API_KEY is not set in .env file"
        exit 1
    fi
    
    print_success "Environment variables are configured"
}

# Function to create necessary directories
setup_directories() {
    print_status "Creating necessary directories..."
    mkdir -p data/pdfs data/vectorstore logs
    print_success "Directories created"
}

# Function to build Docker image
build_docker() {
    print_status "Building Docker image..."
    docker-compose build
    print_success "Docker image built successfully"
}

# Function to run Docker container (create and start)
run_docker() {
    print_status "Running Docker container (create and start)..."
    docker-compose up -d
    print_success "Docker container started in detached mode"
    
    # Wait for container to be ready
    print_status "Waiting for container to be ready..."
    sleep 10
    
    # Check if container is running
    if docker-compose ps | grep -q "Up"; then
        print_success "Container is running"
    else
        print_error "Container failed to start"
        exit 1
    fi
}

# Function to start Docker container (start existing stopped container)
start_docker() {
    print_status "Starting existing Docker container..."
    docker-compose start
    print_success "Docker container started"
    
    # Wait for container to be ready
    print_status "Waiting for container to be ready..."
    sleep 5
    
    # Check if container is running
    if docker-compose ps | grep -q "Up"; then
        print_success "Container is running"
    else
        print_error "Container failed to start"
        exit 1
    fi
}

# Function to ingest PDFs
ingest_pdfs() {
    local force_flag=""
    if [ "$1" = "--force" ]; then
        force_flag="--force"
    fi
    
    print_status "Ingesting PDFs from data/pdfs/ directory..."
    
    # Check if PDFs exist
    PDF_COUNT=$(find data/pdfs -name "*.pdf" 2>/dev/null | wc -l)
    if [ $PDF_COUNT -eq 0 ]; then
        print_warning "No PDF files found in data/pdfs/"
        print_status "You can add PDF files to data/pdfs/ directory and run ingestion later with:"
        print_status "docker-compose exec chat-with-pdf python scripts/ingest_pdfs.py default"
        return
    fi
    
    print_status "Found $PDF_COUNT PDF file(s). Starting ingestion..."
    docker-compose exec -T chat-with-pdf python scripts/ingest_pdfs.py default $force_flag
    
    if [ $? -eq 0 ]; then
        print_success "PDF ingestion completed successfully"
    else
        print_error "PDF ingestion failed"
        exit 1
    fi
}

# Function to show logs (follow mode)
show_logs() {
    print_status "Showing Docker container logs (follow mode)..."
    print_status "Press Ctrl+C to exit log viewing"
    echo ""
    docker-compose logs -f --tail=50
}

# Function to get Docker logs (one-time view)
get_docker_logs() {
    print_status "Getting Docker container logs..."
    docker-compose logs --tail=100
    echo ""
    print_status "For live logs, use: $0 logs"
}

# Function to check health
check_health() {
    print_status "Checking container health..."
    
    # Check health endpoint
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        print_success "Health check passed"
        echo ""
        print_status "Application URLs:"
        echo "  - API Documentation: http://localhost:8000/docs"
        echo "  - Health Check: http://localhost:8000/health"
        echo "  - Ask Question: http://localhost:8000/ask"
    else
        print_warning "Health check failed - container may still be starting"
        print_status "Try again in a few moments or check logs"
    fi
}

# Main execution
main() {
    echo "=================================="
    echo "Chat with PDF - Docker Setup"
    echo "=================================="
    echo ""
    
    # Step 1: Check Docker
    check_docker
    
    # Step 2: Check environment
    check_environment
    
    # Step 3: Setup directories
    setup_directories
    
    # Step 4: Build Docker image
    build_docker
    
    # Step 5: Run Docker container
    run_docker
    
    # Step 6: Ingest PDFs
    # ingest_pdfs
    
    # Step 7: Check health
    sleep 5
    check_health
    
    echo ""
    print_success "Setup completed successfully!"
    echo ""
    print_status "Useful commands:"
    echo "  - View logs: docker-compose logs -f"
    echo "  - Stop container: docker-compose down"
    echo "  - Restart: docker-compose restart"
    echo "  - Shell access: docker-compose exec chat-with-pdf /bin/bash"
    echo ""
    print_status "To view logs now, run: $0 logs"
}

# Handle command line arguments
case "${1:-}" in
    "build")
        check_docker
        build_docker
        ;;
    "run")
        check_docker
        check_environment
        run_docker
        ;;
    "start")
        check_docker
        start_docker
        ;;
    "logs")
        show_logs
        ;;
    "get-logs")
        get_docker_logs
        ;;
    "health")
        check_health
        ;;
    "ingest")
        ingest_pdfs "$2"
        ;;
    "help")
        echo "Usage: $0 [COMMAND]"
        echo ""
        echo "Commands:"
        echo "  build       Build Docker image only"
        echo "  run         Run Docker container (create and start)"
        echo "  start       Start existing stopped container"
        echo "  logs        Show live Docker logs (follow mode)"
        echo "  get-logs    Get Docker logs (one-time view)"
        echo "  health      Check container health"
        echo "  ingest      Ingest PDFs from data/pdfs/ (use 'ingest --force' to force append)"
        echo "  help        Show this help message"
        echo ""
        echo "Examples:"
        echo "  $0              # Complete setup (build, run, ingest)"
        echo "  $0 build        # Build image only"
        echo "  $0 run          # Run container"
        echo "  $0 get-logs     # View recent logs"
        echo "  $0 logs         # Follow live logs"
        ;;
    *)
        main
        ;;
esac