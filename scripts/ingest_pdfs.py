#!/usr/bin/env python3
"""Script to ingest PDF files into the vector database."""

from src.services.pdf_processor import pdf_processor_service
from src.services.vector_store import vector_store_service
from src.core.config import settings, ensure_directories
import argparse
import asyncio
import sys
import os
from pathlib import Path
import structlog

# Add the project root directory to the path BEFORE importing src modules
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

# Now we can safely import from src

# Setup logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


async def ingest_single_file(file_path: str, force: bool = False) -> bool:
    """Ingest a single PDF file."""
    try:
        logger.info("Ingesting single PDF", file_path=file_path, force=force)

        if not os.path.exists(file_path):
            logger.error("File not found", file_path=file_path)
            return False

        if not file_path.lower().endswith('.pdf'):
            logger.error("File is not a PDF", file_path=file_path)
            return False

        result = await pdf_processor_service.ingest_single_pdf(file_path, force=force)

        if result["success"]:
            logger.info(
                "Successfully ingested PDF",
                file_path=file_path,
                chunks=result["chunks_processed"]
            )
            print(f"✅ Successfully ingested {result['file_name']}")
            print(f"   📄 Processed {result['chunks_processed']} chunks")
            return True
        else:
            # Check if it's an existing data warning
            if 'existing_files' in result:
                print(f"\n⚠️  {result['message']}")
                print(f"\n📊 Current vector store status:")
                print(f"   📁 Files: {len(result['existing_files'])}")
                print(f"   📄 Documents: {result['existing_documents']}")
                print(f"\n📂 Existing files:")
                for file_name in result['existing_files']:
                    print(f"   • {file_name}")
                print(f"\n💡 Options:")
                print(f"   1. Clear existing data: python scripts/ingest_pdfs.py clear")
                print(
                    f"   2. Force append: python scripts/ingest_pdfs.py file \"{file_path}\" --force")
                return False
            else:
                logger.error(
                    "Failed to ingest PDF",
                    file_path=file_path,
                    message=result["message"]
                )
                print(f"❌ Failed to ingest PDF: {result['message']}")
                return False

    except Exception as e:
        logger.error("Error ingesting PDF", file_path=file_path, error=str(e))
        print(f"❌ Error ingesting PDF: {str(e)}")
        return False


async def ingest_directory(directory_path: str, force: bool = False) -> int:
    """Ingest all PDF files from a directory."""
    try:
        logger.info("Ingesting PDFs from directory",
                    directory_path=directory_path, force=force)

        if not os.path.exists(directory_path):
            logger.error("Directory not found", directory_path=directory_path)
            print(f"❌ Directory not found: {directory_path}")
            return 0

        total_chunks = await pdf_processor_service.process_pdf_directory(directory_path, force=force)

        logger.info(
            "Directory ingestion completed",
            directory_path=directory_path,
            total_chunks=total_chunks
        )

        print(f"✅ Directory ingestion completed")
        print(f"   📄 Total chunks processed: {total_chunks}")
        return total_chunks

    except ValueError as e:
        # Handle existing data error
        error_msg = str(e)
        print(f"\n⚠️  {error_msg}")

        # Show existing files
        existing_files = pdf_processor_service.list_processed_files()
        print(f"\n📂 Existing files:")
        for file_info in existing_files:
            print(
                f"   • {file_info['file_name']}: {file_info['chunk_count']} chunks")

        print(f"\n💡 Options:")
        print(f"   1. Clear existing data: python scripts/ingest_pdfs.py clear")
        print(
            f"   2. Force append: python scripts/ingest_pdfs.py directory \"{directory_path}\" --force")
        return 0

    except Exception as e:
        logger.error("Error ingesting directory",
                     directory_path=directory_path, error=str(e))
        print(f"❌ Error ingesting directory: {str(e)}")
        return 0


async def clear_database():
    """Clear all documents from the vector database."""
    try:
        logger.info("Clearing vector database")

        success = await pdf_processor_service.clear_all_documents()

        if success:
            logger.info("Vector database cleared successfully")
        else:
            logger.error("Failed to clear vector database")

        return success

    except Exception as e:
        logger.error("Error clearing database", error=str(e))
        return False


async def show_stats():
    """Show database statistics."""
    try:
        logger.info("Getting database statistics")

        # Get vector store stats
        vector_stats = vector_store_service.get_stats()

        # Get processed files
        processed_files = pdf_processor_service.list_processed_files()

        print("\n📊 Vector Database Statistics")
        print("=" * 50)
        print(f"📄 Total documents: {vector_stats['total_documents']}")
        print(f"🔢 Index size: {vector_stats['index_size']}")
        print(f"📐 Dimension: {vector_stats['dimension']}")
        print(f"💾 Storage path: {vector_stats['storage_path']}")
        print(
            f"🗃️  Has data: {'Yes' if vector_stats['has_existing_data'] else 'No'}")

        print(f"\n📂 Processed Files ({len(processed_files)} files)")
        print("=" * 50)
        if processed_files:
            total_chunks = sum(f['chunk_count'] for f in processed_files)
            for file_info in processed_files:
                print(
                    f"• {file_info['file_name']}: {file_info['chunk_count']} chunks")
            print(f"\n📈 Total chunks across all files: {total_chunks}")
        else:
            print("📭 No files have been processed yet.")

        print(f"\n⚙️  Configuration")
        print("=" * 50)
        print(f"📁 PDF storage path: {settings.pdf_storage_path}")
        print(f"📏 Chunk size: {settings.chunk_size}")
        print(f"🔄 Chunk overlap: {settings.chunk_overlap}")

    except Exception as e:
        logger.error("Error getting statistics", error=str(e))
        print(f"❌ Error getting statistics: {str(e)}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Ingest PDF files into the Chat with PDF vector database"
    )

    subparsers = parser.add_subparsers(
        dest="command", help="Available commands")

    # File ingestion command
    file_parser = subparsers.add_parser(
        "file", help="Ingest a single PDF file")
    file_parser.add_argument("path", help="Path to the PDF file")
    file_parser.add_argument("--force", action="store_true",
                             help="Force ingestion even if data exists")

    # Directory ingestion command
    dir_parser = subparsers.add_parser(
        "directory", help="Ingest all PDFs from a directory")
    dir_parser.add_argument(
        "path", help="Path to the directory containing PDFs")
    dir_parser.add_argument("--force", action="store_true",
                            help="Force ingestion even if data exists")

    # Default directory ingestion command
    default_parser = subparsers.add_parser(
        "default", help="Ingest PDFs from default directory")
    default_parser.add_argument("--force", action="store_true",
                                help="Force ingestion even if data exists")

    # Clear command
    clear_parser = subparsers.add_parser(
        "clear", help="Clear all documents from database")

    # Stats command
    stats_parser = subparsers.add_parser(
        "stats", help="Show database statistics")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Ensure directories exist
    ensure_directories()

    async def run_command():
        if args.command == "file":
            force = getattr(args, 'force', False)
            success = await ingest_single_file(args.path, force=force)
            sys.exit(0 if success else 1)

        elif args.command == "directory":
            force = getattr(args, 'force', False)
            total_chunks = await ingest_directory(args.path, force=force)
            sys.exit(0 if total_chunks > 0 else 1)

        elif args.command == "default":
            force = getattr(args, 'force', False)
            total_chunks = await ingest_directory(settings.pdf_storage_path, force=force)
            sys.exit(0 if total_chunks > 0 else 1)

        elif args.command == "clear":
            # Show current stats first
            stats = vector_store_service.get_stats()
            existing_files = pdf_processor_service.list_processed_files()

            if not stats['has_existing_data']:
                print("📭 Vector store is already empty.")
                sys.exit(0)

            print(f"\n📊 Current vector store contains:")
            print(f"   📄 Documents: {stats['total_documents']}")
            print(f"   📁 Files: {len(existing_files)}")
            print(f"\n📂 Files to be deleted:")
            for file_info in existing_files:
                print(
                    f"   • {file_info['file_name']}: {file_info['chunk_count']} chunks")

            print(
                f"\n⚠️  Are you sure you want to clear all documents? This cannot be undone.")
            confirm = input("Type 'yes' to confirm: ").lower().strip()

            if confirm == "yes":
                success = await clear_database()
                if success:
                    print("✅ Successfully cleared all documents from vector store")
                else:
                    print("❌ Failed to clear vector store")
                sys.exit(0 if success else 1)
            else:
                print("🚫 Operation cancelled.")
                sys.exit(0)

        elif args.command == "stats":
            await show_stats()
            sys.exit(0)

    # Run the async command
    asyncio.run(run_command())


if __name__ == "__main__":
    main()
