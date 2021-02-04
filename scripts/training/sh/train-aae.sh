pipenv run python scripts/training/py/train-adversarial-autoencoder.py \
  --input-dataset data/filtered/lowercase \
  --input-dataset data/filtered/uppercase \
  --input-dataset data/filtered/numbers \
  --output-folder models/aae-cs-32-dec-2 \
  --encoder-hyperparameter-file tmp/aae-encoder.json \
  --decoder-hyperparameter-file tmp/aae-decoder-2.json \
  --discr-hyperparameter-file tmp/aae-discriminator.json \
  --steps-per-epoch 5000 \
  --epochs 5 \
  --batch-size 96 \
  --lr-shrink-factor 0.9\
  --code-size 32
  #--reuse-model \
  #--optimizer-hyperparameters tmp/aae-optimizer.json
  