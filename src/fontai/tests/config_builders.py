import pytest


ingestion_config_str = """
scrappers:
- class: LocalScrapper
  kwargs: 
    folder: src/fontai/tests/data/ingestion/input
output_path: src/fontai/tests/data/ingestion/output
"""

processing_config_str = """
input_path: src/fontai/tests/data/ingestion/output
output_record_class: {output_record_class}
output_path: src/fontai/tests/data/preprocessing/output/{output_folder}
output_array_size: 64
max_output_file_size: 64
font_extraction_size: 100
canvas_size: 500
canvas_padding: 100
"""

predictor_config_str = """
input_record_class: {in_type_class}
input_path: src/fontai/tests/data/preprocessing/output/{input_folder}
output_path: src/fontai/tests/data/prediction/output/scored-{output_folder}
model_path: src/fontai/tests/data/prediction/models/{model_folder}
charset: lowercase
training:
  {batch_size}
  epochs: 10
  steps_per_epoch: 10
  optimizer:
    class: Adam
  loss:
    class: CategoricalCrossentropy
  metrics:
  - accuracy
model:
  {model}
"""

sequential_model = """
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
        units: 26
        activation: softmax
"""

char_saae = """
  class: CharStyleSAAE
  kwargs:
    image_encoder:
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
    full_encoder:
      class: Sequential
      kwargs:
        layers:
        - class: Input
          kwargs:
            shape:
            - 36
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
            - 36
        - class: Dense
          kwargs: 
            units: 4096
        - class: Reshape
          kwargs:
            target_shape:
            - 64
            - 64
            - 1
    prior_discriminator:
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


font_saae = """
  class: PureFontStyleSAAE
  kwargs:
    image_encoder:
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
    full_encoder:
      class: Sequential
      kwargs:
        layers:
        - class: Input
          kwargs:
            shape:
            - 36
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
            - 36
        - class: Dense
          kwargs: 
            units: 4096
        - class: Reshape
          kwargs:
            target_shape:
            - 64
            - 64
            - 1
    prior_discriminator:
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


def full_processing_config_str(output_record_class):

  keys = ("output_record_class", "output_folder")

  cases = {
    "LabeledChar": ("LabeledChar", "chars"),
    "LabeledFont": ("LabeledFont", "fonts")
  }

  return processing_config_str.format(**{key: value for key,value in zip(keys,cases[output_record_class])})


def full_prediction_config_str(input_record_class, model):

  keys = ("in_type_class", "input_folder", "output_folder", "model_folder", "model", "batch_size")

  model_strs = {
    "Sequential": sequential_model,
    "CharStyleSAAE": char_saae,
    "FontStyleSAAE": font_saae
  }

  cases  = {
    "LabeledChar": ("LabeledChar", "chars", "chars", model, model_strs[model], "batch_size: 32"),
    "LabeledFont": ("LabeledFont", "fonts", "fonts", model, model_strs[model], "batch_size: 2")
  }

  return predictor_config_str.format(**{key: value for key,value in zip(keys,cases[input_record_class])})
