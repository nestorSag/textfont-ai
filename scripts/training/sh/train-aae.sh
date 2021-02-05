pipenv run python scripts/training/py/train-adversarial-autoencoder.py \
  --input-dataset data/filtered/lowercase \
  --input-dataset data/filtered/uppercase \
  --input-dataset data/filtered/numbers \
  --output-folder models/aae-1 \
  --encoder-hyperparameter-file tmp/aae-encoder.json \
  --decoder-hyperparameter-file tmp/aae-decoder.json \
  --discr-hyperparameter-file tmp/aae-discriminator.json \
  --steps-per-epoch 1000 \
  --epochs 1\
  --batch-size 96 \
  --lr-shrink-factor 0.95\
  --code-size 16 \
  --reconstruction-loss-weight 0.001 \
  --reuse-model
  #--optimizer-hyperparameters tmp/aae-optimizer.json
  