pipenv run python scripts/training/py/train-adversarial-autoencoder.py \
  --input-dataset data/filtered/lowercase \
  --input-dataset data/filtered/uppercase \
  --input-dataset data/filtered/numbers \
  --output-folder models/o-aae-32-h \
  --encoder-hyperparameter-file tmp/aae-encoder.json \
  --decoder-hyperparameter-file tmp/aae-decoder.json \
  --discr-hyperparameter-file tmp/aae-discriminator.json \
  --steps-per-epoch 100 \
  --epochs 10\
  --batch-size 96 \
  --lr-shrink-factor 1\
  --code-size 16 \
  --reconstruction-loss-weight 0.00001 \
  --single-char h
  #--optimizer-hyperparameters tmp/aae-optimizer.json
  