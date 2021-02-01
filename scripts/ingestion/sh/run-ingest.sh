#!/bin/bash
pipenv run python scripts/ingestion/py/ingest.py \
  --output-folder data/raw 
  
pipenv run python scripts/ingestion/py/split-google-zipfile.py \
  --input-file data/raw/google/zip/fonts-master \
  --output-folder data/raw/google/zip/split && rm data/raw/google/zip/fonts-master

mkdir data/raw/ingested && mv data/raw/*.zip data/raw/ingested