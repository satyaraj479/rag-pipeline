#!/usr/bin/env python
"""CLI: python ingest.py <file_or_directory>"""
import sys

from dotenv import load_dotenv

from pipeline import ingest

load_dotenv()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ingest.py <file_or_directory>")
        sys.exit(1)
    ingest(sys.argv[1])
