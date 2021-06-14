from fontai.pipeline.stages import Predictor

SEQUENTIAL_PREDICTOR_CONFIG = """
input_path: src/tests/data/preprocessing/output
output_path: src/tests/data/training/output
model_path: src/tests/data/training/model
training:
  batch_size: 32
  epochs: 10
  steps_per_epoch: 10
  optimizer:
    class: Adam
  loss:
    class: CategoricalCrossentropy
  metrics:
  - accuracy
model:
  class: Sequential
  kwargs:
    layers:
    - class: Dense
      kwargs: 
        units: 10
        activation: elu
"""


AAE_PREDICTOR_CONFIG = """
input_path: src/tests/data/preprocessing/output
output_path: src/tests/data/training/output
model_path: src/tests/data/training/model
training:
  batch_size: 32
  epochs: 10
  steps_per_epoch: 10
  optimizer:
    class: Adam
  loss:
    class: CategoricalCrossentropy
  metrics:
  - accuracy
model:
  class: SAAE
  kwargs:
    encoder:
      class: Sequential
      kwargs:
        layers:
        - class: Dense
          kwargs: 
            units: 10
            activation: elu
    decoder:
      class: Sequential
      kwargs:
        layers:
        - class: Dense
          kwargs: 
            units: 10
            activation: elu
    discriminator:
      class: Sequential
      kwargs:
        layers:
        - class: Dense
          kwargs: 
            units: 10
            activation: elu
    code_dim: 10,
    reconstruction_loss_weight:float: 0.5,
    input_dim: 64,
    n_classes: 62,
    prior_batch_size: 32
"""

def test_predictor():
  config = Predictor.parse_config_str(SEQUENTIAL_PREDICTOR_CONFIG)
  predictor = Predictor.from_config_object(config)





