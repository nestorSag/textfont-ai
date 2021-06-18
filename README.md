
# ML pipeline for text font generation [under development]

This package contains the codebase of an end-to-end ML pipeline to train generative models for text font generation from open source font files scrapped from the internet. All of the ML stages, from ingestion to model training are designed to be run from configuration files and can be concatenated through a Pipeline class; correct configuration semantics for each one are enforced by the `config` submodule. A brief overview of the submodules is below:

    `config`: Parsers that map YAML configuration files to object wrappers for all the dependencies that the corresponding ML stage needs to be executed either in batch or streaming mode; it makes heavy use of the `strictyaml` and `pydantic` libraries.

    `io`: Contains logic pertaining to data interfaces between ML stages and storage and to other ML stages; this comprises supported Tensorflow record formats, file formats, storage media and file reader and writer classes.

    `pipeline`: High-level ML stage executors are here, as well as the basic interfaces that each one has to implement.

    `prediction`: Contains the logic for preprocessing Tensorflow examples and get them ready for model training or scoring; also defines custom model architectures and callback functions for training.

    `preprocessing`: Contains the logic for mapping zipped font files to labeled character images ready to be ingested for model training.

## Organisation

The codebase has 2 main parts, a core python library, `fontai`, in the `src` folder, and a set of scripts built on top of it in the `scripts` folder. The `fontai` package contains the core logic necessary to run the different project's stages, while the `scripts` act as wrappers for `fontai` calls.

## ML projects as code

As this is also an attempt of a production-grade ML pipeline system, execution relies entirely on predefined YAML configuration files. There shoudln't even be a need to star a Python console, instead it is enough to point the corresponding execution script in the `scripts` folder to the correct YAML configuration file. Configuration definitions are flexible enough to enable defining arbitrary Sequential Keras models and changing every parameter in the preprocessing stage, etc. Non-sequential model architectures can be used as well provided they are defined in the `fontai.prediction.models` module.

## Stage summaries

### Ingestion (urls -> zipped font files)

The data is ingested by web scrapping multiple websites for free font files. This results in 80,400 font zip files from which a bit over 125,000 different fonts could be used at the time it was first tried. Note that some of the original scrappers (in `fontai.io.scrappers`) have stopped working as of June 2021 due to changes in the structure of the source websties; some work might be required to make them work again.


### Preprocessing (zippedfont files -> Tensorflow record files)

Font files are unzipped and converted from `ttf` or `otf` to 64x64 `png` images for each alphanumeric character in each font using Apache Beam, and are then batched together into Tensorflow record files for model consumption.

### Training (Tensorflow record files -> trained model)

The models are ran using Tensorflow's Keras framework.


##Getting started

### Installation

clone the repository and run `pip install src/fontai`

### Execution

Open source Google fonts can be used for a start; the main zip file includes around 1k fonts. create the following configuration YAML files

```yaml
#ingestion-config.yaml
scrappers:
- class: GoogleFontsScrapper
output_path: data/raw
```

```yaml
#preprocessing-config.yaml
input_path: data/raw
output_path: data/preprocessed
output_array_size: 64
max_output_file_size: 64
font_extraction_size: 100
canvas_size: 500
canvas_padding: 100
beam_cmd_line_args:
-  --runner
-  DirectRunner
- --direct_num_workers
- 2
```

```yaml
#training-config.yaml
# this model is so small it's probably useless, but it's a good example; look at config/parameters for more realistic model architectures
input_path: data/preprocessed
output_path: data/scored
model_path: models/ #model will be saved here
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
```
Then run 
```py
from fontai.pipeline.stages import FontIngestion, LabeledExampleExtractor, Predictor

FontIngestion.run_from_config_file("path/to/ingestion-config.yaml")
LabeledExampleExtractor.run_from_config_file("path/to/preprocessing-config.yaml")
Predictor.fit_from_config_file("path/to/training-config.yaml") #this will fit the model; to do batch scoring using the fitted model you can run Predictor.run_from_config_file("path/to/training-config.yaml", load_from_model_path=True) immediately after.
```

This can also be run in a single pipeline object, but then an overarching configuration file is required:

```yaml
#pipeline-config.yaml
stages:
- class: FontIngestion
  yaml_config_path: path/to/ingestion-config.yaml
- class: LabeledExamplesxtractor
  yaml_config_path: path/to/preprocessing-config.yaml
- class: Predictor
  yaml_config_path: path/to/training-config.yaml

```

Then run:

```py
from fontai.pipeline.pipeline import Pipeline

# this will train the model
Pipeline.fit_from_config_file("path/to/pipeline-config.yaml")
```
That's it. Virtually any parameter in the entire pipeline can be changed through YAML files

