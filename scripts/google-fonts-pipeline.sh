#!/bin/bash

# This script runs a pipeline using google fonts, with around 4k examples.

fontairun --stage ingestion \
  --config-file config/parameters/google-fonts-examples/ingestion.yaml

fontairun --stage preprocessing \
  --config-file config/parameters/google-fonts-examples/preprocessing.yaml

# if --fit is not passed below, the model simply scores the input data and outputs scored examples to the provided location.
fontairun --stage scoring \
  --fit \
  --config-file config/parameters/google-fonts-examples/training.yaml

fontairun --stage deployment \
  --config-file config/parameters/google-fonts-examples/deployment.yaml