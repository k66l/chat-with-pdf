#!/bin/bash

# Chat with PDF - Docker Build and Run Script
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

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS] COMMAND"
    echo ""
    echo "Commands:"
    echo "  build       Build the Docker image"
    echo "  run         Run the container"
    echo "  start       Start the container (detached)"
    echo "  stop        Stop the container"
    echo "  restart     Restart the container"
    echo "  logs        Show container logs"
    echo "  shell       Open shell in running container"
    echo "  clean       Remove containers and images"
    echo "  setup       Complete setup (build + run)"
    echo "  ingest      Ingest PDFs into the running container"
    echo "  health      Check container health"
    echo ""
    echo "Options:"
    echo "  -h, --help  Show this help message"
    echo "  -f, --force Force rebuild (for build command) or force append (for ingest)"
    echo "  -d, --detach Run in detached mode (for run command)"
    echo ""
    echo "Examples:"
    echo "  $0 build"
    echo "  $0 run"
    echo "  $0 setup"
    echo "  $0 ingest"
}

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
            print_warning "Then run: $0 setup"
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
build_image() {
    print_status "Building Docker image..."
    
    local force_rebuild=false
    if [[ "$1" == "--force" ]] || [[ "$1" == "-f" ]]; then
        force_rebuild=true
    fi
    
    if [ "$force_rebuild" = true ]; then
        print_status "Force rebuilding image..."
        docker-compose build --no-cache
    else
        docker-compose build
    fi
    
    print_success "Docker image built successfully"
}

# Function to run container
run_container() {
    print_status "Starting container..."
    
    local detach=false
    if [[ "$1" == "--detach" ]] || [[ "$1" == "-d" ]]; then
        detach=true
    fi
    
    if [ "$detach" = true ]; then
        docker-compose up -d
        print_success "Container started in detached mode"
        print_status "Use '$0 logs' to view logs"
    else
        docker-compose up
    fi
}

# Function to stop container
stop_container() {
    print_status "Stopping container..."
    docker-compose down
    print_success "Container stopped"
}

# Function to show logs
show_logs() {
    print_status "Showing container logs..."
    docker-compose logs -f
}

# Function to open shell in container
open_shell() {
    print_status "Opening shell in container..."
    docker-compose exec chat-with-pdf /bin/bash
}

# Function to clean up
clean_up() {
    print_status "Cleaning up Docker resources..."
    docker-compose down --rmi all --volumes --remove-orphans
    print_success "Cleanup completed"
}

# Function to check health
check_health() {
    print_status "Checking container health..."
    
    if docker-compose ps | grep -q "Up"; then
        print_success "Container is running"
        
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
        fi
    else
        print_error "Container is not running"
        exit 1
    fi
}

# Function to ingest PDFs
ingest_pdfs() {
    print_status "Ingesting PDFs..."
    
    # Check if container is running
    if ! docker-compose ps | grep -q "Up"; then
        print_error "Container is not running. Start it first with: $0 start"
        exit 1
    fi
    
    # Check if PDFs exist
    PDF_COUNT=$(find data/pdfs -name "*.pdf" 2>/dev/null | wc -l)
    if [ $PDF_COUNT -eq 0 ]; then
        print_warning "No PDF files found in data/pdfs/"
        print_status "Add PDF files to data/pdfs/ directory and run this command again"
        exit 1
    fi
    
    # Check if --force flag is provided
    local force_flag=""
    if [[ "$2" == "--force" ]] || [[ "$2" == "-f" ]]; then
        force_flag="--force"
        print_status "Using force flag to append to existing data..."
    fi
    
    print_status "Found $PDF_COUNT PDF file(s). Starting ingestion..."
    if [ -n "$force_flag" ]; then
        docker-compose exec -w /app chat-with-pdf python scripts/ingest_pdfs.py default --force
    else
        docker-compose exec -w /app chat-with-pdf python scripts/ingest_pdfs.py default
    fi
    
    print_success "PDF ingestion completed"
}

# Function for complete setup
complete_setup() {
    print_status "Starting complete setup..."
    
    check_docker
    check_environment
    setup_directories
    build_image
    run_container --detach
    
    print_success "Setup completed!"
    echo ""
    print_status "Next steps:"
    echo "  1. Add PDF files to data/pdfs/ directory"
    echo "  2. Run: $0 ingest"
    echo "  3. Access the API at: http://localhost:8000/docs"
    echo ""
    print_status "Useful commands:"
    echo "  - View logs: $0 logs"
    echo "  - Check health: $0 health"
    echo "  - Stop container: $0 stop"
    echo "  - Restart: $0 restart"
}

# Main script logic
case "${1:-}" in
    "build")
        check_docker
        build_image "$2"
        ;;
    "run")
        check_docker
        check_environment
        run_container "$2"
        ;;
    "start")
        check_docker
        check_environment
        run_container --detach
        ;;
    "stop")
        stop_container
        ;;
    "restart")
        stop_container
        run_container --detach
        ;;
    "logs")
        show_logs
        ;;
    "shell")
        open_shell
        ;;
    "clean")
        clean_up
        ;;
    "setup")
        complete_setup
        ;;
    "ingest")
        ingest_pdfs "$@"
        ;;
    "health")
        check_health
        ;;
    "-h"|"--help"|"help"|"")
        show_usage
        ;;
    *)
        print_error "Unknown command: $1"
        show_usage
        exit 1
        ;;
esac 