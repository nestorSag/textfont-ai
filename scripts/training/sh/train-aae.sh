pipenv run python scripts/training/py/train-original-adversarial-autoencoder.py \
  --input-dataset data/filtered/numbers \
  --output-folder models/aae-32-4 \
  --encoder-hyperparameter-file tmp/aae-encoder.json \
  --decoder-hyperparameter-file tmp/aae-decoder.json \
  --discr-hyperparameter-file tmp/aae-discriminator.json \
  --steps-per-epoch 1000 \
  --epochs 250\
  --batch-size 96 \
  --lr-shrink-factor 0.985\
  --code-size 32 \
  --reconstruction-loss-weight 0.0001 \
  --single-char 4
# pipenv run python scripts/training/py/train-original-adversarial-autoencoder.py \
#   --input-dataset data/filtered/lowercase \
#   --output-folder models/o-aae-16-h-zero \
#   --encoder-hyperparameter-file tmp/aae-encoder.json \
#   --decoder-hyperparameter-file tmp/aae-decoder.json \
#   --discr-hyperparameter-file tmp/aae-discriminator.json \
#   --steps-per-epoch 1000 \
#   --epochs 200\
#   --batch-size 96 \
#   --lr-shrink-factor 0.977\
#   --code-size 16 \
#   --reconstruction-loss-weight 0.0000 \
#   --single-char h \
#   --orthogonal-autoencoder
#   #--optimizer-hyperparameters tmp/aae-optimizer.json

# pipenv run python scripts/training/py/train-original-adversarial-autoencoder.py \
#   --input-dataset data/filtered/lowercase \
#   --output-folder models/o-aae-16-h \
#   --encoder-hyperparameter-file tmp/aae-encoder.json \
#   --decoder-hyperparameter-file tmp/aae-decoder.json \
#   --discr-hyperparameter-file tmp/aae-discriminator.json \
#   --steps-per-epoch 1000 \
#   --epochs 150\
#   --batch-size 96 \
#   --lr-shrink-factor 0.977\
#   --code-size 16 \
#   --reconstruction-loss-weight 0.00001 \
#   --single-char h \
#   --orthogonal-autoencoder
#   #--optimizer-hyperparameters tmp/aae-optimizer.json