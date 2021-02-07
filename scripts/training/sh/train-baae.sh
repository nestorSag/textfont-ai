pipenv run python scripts/training/py/train-biadversarial-autoencoder.py \
  --input-dataset data/filtered/lowercase \
  --input-dataset data/filtered/uppercase \
  --input-dataset data/filtered/numbers \
  --output-folder models/o-baae-32 \
  --encoder-hyperparameter-file tmp/aae-encoder.json \
  --decoder-hyperparameter-file tmp/aae-decoder.json \
  --code-discr-hyperparameter-file tmp/aae-discriminator.json \
  --img-discr-hyperparameter-file tmp/aae-img-discriminator.json \
  --steps-per-epoch 1000 \
  --epochs 5\
  --batch-size 96 \
  --lr-shrink-factor 1\
  --code-size 32
    #--optimizer-hyperparameters tmp/aae-optimizer.json
  