#!/bin/bash

# Chat with PDF - PDF Ingestion Script
set -e

echo "📚 Chat with PDF - PDF Ingestion Tool"
echo "======================================"

# Default values
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INGEST_SCRIPT="$SCRIPT_DIR/scripts/ingest_pdfs.py"
VENV_DIR=".venv"

# Check if virtual environment exists
if [ ! -d "$VENV_DIR" ]; then
    echo "❌ Virtual environment not found at $VENV_DIR"
    echo "   Please run ./start.sh first to set up the environment"
    exit 1
fi

# Check if ingest script exists
if [ ! -f "$INGEST_SCRIPT" ]; then
    echo "❌ Ingestion script not found at $INGEST_SCRIPT"
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Set PYTHONPATH to include the project root
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"
echo "🔧 Setting PYTHONPATH to: $SCRIPT_DIR"

# Function to show usage
show_usage() {
    echo ""
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  file <path>          Ingest a single PDF file"
    echo "  directory <path>     Ingest all PDFs from a directory"
    echo "  default              Ingest PDFs from default directory (data/pdfs/)"
    echo "  stats                Show database statistics"
    echo "  clear                Clear all documents from database"
    echo "  help                 Show this help message"
    echo ""
    echo "Options:"
    echo "  --force              Force ingestion even if data exists (for file/directory/default)"
    echo ""
    echo "Examples:"
    echo "  $0 file /path/to/document.pdf"
    echo "  $0 file /path/to/document.pdf --force"
    echo "  $0 directory /path/to/pdf/folder"
    echo "  $0 directory /path/to/pdf/folder --force"
    echo "  $0 default"
    echo "  $0 default --force"
    echo "  $0 stats"
    echo "  $0 clear"
    echo ""
}

# Function to run ingestion command
run_ingest() {
    local command="$1"
    shift
    
    echo "🚀 Running PDF ingestion..."
    echo "   Command: $command"
    
    case "$command" in
        "file")
            if [ -z "$1" ]; then
                echo "❌ Error: File path required"
                echo "   Usage: $0 file <path-to-pdf> [--force]"
                exit 1
            fi
            
            filepath="$1"
            shift
            
            if [ ! -f "$filepath" ]; then
                echo "❌ Error: File not found: $filepath"
                exit 1
            fi
            
            echo "   File: $filepath"
            python "$INGEST_SCRIPT" file "$filepath" "$@"
            ;;
            
        "directory")
            if [ -z "$1" ]; then
                echo "❌ Error: Directory path required"
                echo "   Usage: $0 directory <path-to-directory> [--force]"
                exit 1
            fi
            
            dirpath="$1"
            shift
            
            if [ ! -d "$dirpath" ]; then
                echo "❌ Error: Directory not found: $dirpath"
                exit 1
            fi
            
            echo "   Directory: $dirpath"
            python "$INGEST_SCRIPT" directory "$dirpath" "$@"
            ;;
            
        "default")
            echo "   Using default directory: data/pdfs/"
            python "$INGEST_SCRIPT" default "$@"
            ;;
            
        "stats")
            echo "📊 Database Statistics:"
            python "$INGEST_SCRIPT" stats
            ;;
            
        "clear")
            echo "⚠️  WARNING: This will delete ALL documents from the database!"
            echo "   This action cannot be undone."
            echo ""
            read -p "Are you sure you want to continue? (yes/no): " confirm
            
            if [ "$confirm" = "yes" ]; then
                echo "yes" | python "$INGEST_SCRIPT" clear
                echo "✅ Database cleared successfully"
            else
                echo "❌ Operation cancelled"
                exit 0
            fi
            ;;
            
        "help")
            show_usage
            exit 0
            ;;
            
        *)
            echo "❌ Error: Unknown command: $command"
            show_usage
            exit 1
            ;;
    esac
}

# Check if no arguments provided
if [ $# -eq 0 ]; then
    echo "❌ Error: No command provided"
    show_usage
    exit 1
fi

# Parse command
COMMAND="$1"
shift

# Handle special cases
case "$COMMAND" in
    "-h"|"--help"|"help")
        show_usage
        exit 0
        ;;
esac

# Run the ingestion command
echo ""
run_ingest "$COMMAND" "$@"

# Show completion message
echo ""
echo "✅ Operation completed successfully!"
echo ""
echo "💡 Useful commands:"
echo "   📊 Check status: $0 stats"
echo "   🚀 Start app: ./start.sh"
echo "   🔍 API docs: http://localhost:4200/docs" 