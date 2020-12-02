## upload dataset to cloud storage
gsutil -m cp -r data/raw/* gs://textfont-ai-data/raw/

## run beam test on local
pipenv run python scripts/py/run-beam-preprocessing.py \
  --input-folder=gs://textfont-ai-data/raw_sample/google/zip \
  --output-folder=gs://textfont-ai-data/local-test-output \
  --runner=DirectRunner \
  --project=textfont-ai \
  --region=europe-west2 \
  --job_name=textffont-ai-zip2png \
  --temp-location=gs://textfont-ai-misc/dataflow/temp/

# run beam test on dataflow
pipenv run python scripts/py/run-beam-preprocessing.py \
  --requirements_file=conf/dataflow-preprocessing-requirements.txt \
  --input-folder=gs://textfont-ai-data/raw_sample/dafont/zip \
  --output-folder=gs://textfont-ai-data/cloud-test-output \
  --runner=DataflowRunner \
  --project=textfont-ai \
  --region=europe-west2 \
  --job_name=textfont-ai-zip2png \
  --temp_location=gs://textfont-ai-misc/dataflow/temp/ \
  --staging_location=gs://textfont-ai-misc/dataflow/staging/ \
  --machine_type=n2-standard-2 \
  --max_num_workers=2

# run dataflow on entire datasets
pipenv run python scripts/py/run-beam-preprocessing.py \
  --requirements_file=conf/dataflow-preprocessing-requirements.txt \
  --input-folder=gs://textfont-ai-data/raw \
  --output-folder=gs://textfont-ai-data/processed-pngs \
  --runner=DataflowRunner \
  --project=textfont-ai \
  --region=europe-west2 \
  --job_name=zip-to-png-preprocessing \
  --temp_location=gs://textfont-ai-misc/dataflow/temp/ \
  --staging_location=gs://textfont-ai-misc/dataflow/staging/ \
  --machine_type=n2-standard-4 \
  --max_num_workers=20 \
  --disk_size_gb=50 \
  --num_workers=10
