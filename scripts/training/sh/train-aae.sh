pipenv run python scripts/training/py/train-adversarial-autoencoder.py \
  --input-dataset data/filtered/lowercase \
  --input-dataset data/filtered/uppercase \
  --input-dataset data/filtered/numbers \
  --output-folder models/aae-1 \
  --encoder-hyperparameter-file tmp/aae-encoder.json \
  --decoder-hyperparameter-file tmp/aae-decoder.json \
  --discr-hyperparameter-file tmp/aae-discriminator.json \
  --steps-per-epoch 100 \
  --epochs 2 \
  --batch-size 64 \
  --lr-shrink-factor 0.9
  