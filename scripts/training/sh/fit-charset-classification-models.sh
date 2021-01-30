#!/bin/bash
pipenv run python scripts/training/py/train-classifier.py \
  --input-folder data/train \
  --output-folder models/supervised-model-1-lowercase \
  --padding 0 \
  --charset lowercase \
  --hyperparameter-file tmp/classifier-64-3x3-padded-elu-batchnorm.json \
  --steps-per-epoch 10000 \
  --epochs 10 \
  --validation-folder data/val \
  --batch-size 128 \
  --lr-shrink-factor 0.9
  
pipenv run python scripts/training/py/train-classifier.py \
  --input-folder data/train \
  --output-folder models/supervised-model-1-uppercase \
  --padding 0 \
  --charset uppercase \
  --hyperparameter-file tmp/classifier-64-3x3-padded-elu-batchnorm.json \
  --steps-per-epoch 10000 \
  --epochs 10 \
  --validation-folder data/val \
  --batch-size 128 \
  --lr-shrink-factor 0.9

pipenv run python scripts/training/py/train-classifier.py \
  --input-folder data/train \
  --output-folder models/supervised-model-1-numbers \
  --padding 0 \
  --charset numbers \
  --hyperparameter-file tmp/classifier-64-3x3-padded-elu-batchnorm.json \
  --steps-per-epoch 10000 \
  --epochs 10 \
  --validation-folder data/val \
  --batch-size 128 \
  --lr-shrink-factor 0.9
  