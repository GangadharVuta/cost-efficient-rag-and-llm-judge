import sys
import argparse
import json
from src.rag.pipeline import RAGPipeline

def main():
    parser = argparse.ArgumentParser(description="Cost-Efficient RAG Application CLI")
    subparsers = parser.add_subparsers(dest="command", help="Sub-command to execute")

    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest a PDF, HTML, or MD file/directory")
    ingest_parser.add_argument("path", type=str, help="Path to document file or directory")
    ingest_parser.add_argument("--category", type=str, default="general", help="Document category metadata tag")

    # Query command
    query_parser = subparsers.add_parser("query", help="Execute RAG query")
    query_parser.add_argument("text", type=str, help="Query question string")
    query_parser.add_argument("--top_k", type=int, default=4, help="Top-k chunks to retrieve")
    query_parser.add_argument("--filter_type", type=str, default=None, help="Filter by file_type metadata (pdf, html, markdown)")

    # Status command
    subparsers.add_parser("status", help="Show vector store index status")
    # Clear command
    subparsers.add_parser("clear", help="Clear vector store index")

    args = parser.parse_args()
    pipeline = RAGPipeline()

    if args.command == "ingest":
        import os
        if os.path.isdir(args.path):
            res = pipeline.ingest_directory(args.path, category=args.category)
        else:
            res = pipeline.ingest_file(args.path, category=args.category)
        print("\n--- Ingestion Result ---")
        print(json.dumps(res, indent=2))

    elif args.command == "query":
        meta_filter = {"file_type": args.filter_type} if args.filter_type else None
        res = pipeline.query(query_text=args.text, top_k=args.top_k, metadata_filter=meta_filter)
        print("\n--- RAG Query Result ---")
        print(f"Question: {res['query']}\n")
        print(f"Answer:\n{res['answer']}\n")
        print("Citations:", ", ".join(res['citations']))
        print("\nTelemetry:", json.dumps(res['telemetry'], indent=2))

    elif args.command == "status":
        print("\n--- Vector Store Status ---")
        print(f"Store Type: {pipeline.store_type}")
        print(f"Total Stored Vectors: {pipeline.store.count()}")

    elif args.command == "clear":
        pipeline.store.clear()
        print("\nVector store successfully cleared.")

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
