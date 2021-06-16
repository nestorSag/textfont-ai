from fontai.pipeline.stages import Predictor

SEQUENTIAL_PREDICTOR_CONFIG = """
input_path: src/tests/data/preprocessing/output
output_path: src/tests/data/prediction/output
model_path: src/tests/data/prediction/model
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
    - class: Input
      kwargs:
        shape:
        - 64
        - 64
        - 1
    - class: Flatten
    - class: Dense
      kwargs: 
        units: 10
        activation: elu
    - class: Dense
      kwargs: 
        units: 62
        activation: sigmoid
"""


AAE_PREDICTOR_CONFIG = """
input_path: src/tests/data/preprocessing/output
output_path: src/tests/data/prediction/output
model_path: src/tests/data/prediction/model
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
        - class: Input
          kwargs:
            shape:
            - 64
            - 64
            - 1
        - class: Flatten
        - class: Dense
          kwargs: 
            units: 10
    decoder:
      class: Sequential
      kwargs:
        layers:
        - class: Input
          kwargs:
            shape:
            - 72
        - class: Dense
          kwargs: 
            units: 4096
        - class: Reshape
          kwargs:
            target_shape:
            - 64
            - 64
            - 1
    discriminator:
      class: Sequential
      kwargs:
        layers:
        - class: Input
          kwargs:
            shape:
            - 10
        - class: Dense
          kwargs: 
            units: 1
            activation: sigmoid
    reconstruction_loss_weight: 0.5
    prior_batch_size: 32
"""

SAVED_PREDICTOR_CONFIG = """
input_path: src/tests/data/preprocessing/output
output_path: src/tests/data/prediction/output
model_path: src/tests/data/prediction/model
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
  path: src/tests/data/prediction/model
  custom_class: SAAE
"""

@pytest.mark.parametrize("config_string", [SEQUENTIAL_PREDICTOR_CONFIG, AAE_PREDICTOR_CONFIG, SAVED_PREDICTOR_CONFIG])
def test_predictor(config_string):
  config = Predictor.parse_config_str(config_string)
  Predictor.fit_from_config_object(config)
  Predictor.fit_from_config_object(config, load_from_model_path = True)

  assert True








