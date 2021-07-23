#!/bin/bash

# This script runs a pipeline using google fonts, with around 4k examples.

fontairun --stage ingestion \
  --config-file config/parameters/google-fonts-examples/ingestion.yaml

fontairun --stage preprocessing \
  --config-file config/parameters/google-fonts-examples/preprocessing.yaml

## remove corrupted or malformed character images by training a classification model and filtering misclassified chars. Corrupted characters are sometimes parsed but displayed as rectangles for some reason.
fontairun --stage scoring \
  --fit \
  --config-file config/parameters/google-fonts-examples/supervised-training.yaml

# if --fit is not passed, the model simply scores the input data and outputs scored examples to the provided location.
fontairun --stage scoring \
  --load-model \
  --config-file config/parameters/google-fonts-examples/supervised-training.yaml

fontairun --stage scoring \
  --fit \
  --config-file config/parameters/google-fonts-examples/generative-training.yaml

fontairun --stage deployment \
  --config-file config/parameters/google-fonts-examples/deployment.yaml