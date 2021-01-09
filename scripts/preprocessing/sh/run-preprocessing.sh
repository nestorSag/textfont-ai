#!/bin/bash
## scrap websites to get font zips 

## run beam test on local
# pipenv run python scripts/py/run-font-preprocessing.py \
#   --input-folder=gs://textfont-ai-data/raw_sample/google/zip \
#   --output-folder=gs://textfont-ai-data/local-test-output \
#   --runner=DirectRunner \
#   --project=textfont-ai \
#   --region=europe-west2 \
#   --job_name=textffont-ai-zip2png \
#   --temp-location=gs://textfont-ai-misc/dataflow/temp/

# create main library's tarball
pipenv run python src/setup.py sdist

# run dataflow on entire datasets and store it in different folder
pipenv run python scripts/preprocessing/py/preprocessing.py \
  --requirements_file=conf/dataflow-preprocessing-requirements.txt \
  --input-folder=gs://textfont-ai-data/raw \
  --output-folder=gs://textfont-ai-data/processed/tfr/64\
  --runner=DataflowRunner \
  --project=textfont-ai \
  --region=europe-west2 \
  --job_name=zip-to-npy-preprocessing \
  --temp_location=gs://textfont-ai-misc/dataflow/temp/ \
  --staging_location=gs://textfont-ai-misc/dataflow/staging/ \
  --machine_type=n2-standard-4 \
  --max_num_workers=18 \
  --disk_size_gb=50 \
  --num_workers=3 \
  --autoscaling_algorithm=THROUGHPUT_BASED \
  --extra_package dist/$(ls dist/)
