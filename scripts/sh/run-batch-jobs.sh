## upload dataset to cloud storage
gsutil -m cp -r data/raw/* gs://textfont-ai-data/raw/

## run dataflow test
pipenv run python scripts/py/run-beam-processor.py \
  --input-folder=gs://textfont-ai-data/raw_sample/dafont/zip \
  --output-folder=gs://textfont-ai-data/test-output \
  --runner=DirectRunner \
  --project=textfont-ai \
  --region=europe-west2 \
  --job_name=textffont-ai-zip2png \
  --temp-location=gs://textfont-ai-misc/dataflow-temptemp/