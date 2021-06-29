
# ML pipeline for text font generation [under development]

This package contains the codebase of an end-to-end ML pipeline to train generative models for text font generation from open source font files scrapped from the internet. All of the ML stages, from ingestion to model training are designed to be run from configuration files and can be concatenated through a Pipeline class; correct configuration semantics for each one are enforced by the `config` submodule. A brief overview of the submodules is below.

As this is also an attempt of a production-grade ML pipeline system, execution relies entirely on predefined YAML configuration files. There shoudln't even be a need to star a Python console, instead it is enough to point the corresponding execution script in the `scripts` folder to the correct YAML configuration file. Configuration definitions are flexible enough to enable defining arbitrary Sequential Keras models and changing every parameter in the preprocessing stage, etc. Non-sequential model architectures can be used as well provided they are defined in the `fontai.prediction.models` module.

## Organisation

The codebase has 2 main parts, a core python library, `fontai`, in the `src` folder, and a set of scripts built on top of it in the `scripts` folder. The `fontai` package contains the core logic necessary to run the different project's stages, while the `scripts` act as wrappers for `fontai` calls.

## Stage summaries

### Ingestion (urls -> zipped font files)

The data is ingested by web scrapping multiple websites for free font files. Some of the original scrappers (in `fontai.io.scrappers`) have stopped working as of June 2021 due to changes in the structure of the source websties.


### Preprocessing (zippedfont files -> Tensorflow record files)

Font files are unzipped and converted from `ttf` or `otf` to k x k `png` images for each alphanumeric character in each font using Apache Beam, and are then batched together into Tensorflow record files for model consumption.

### Training (Tensorflow record files -> trained model)

The models are ran using Tensorflow's Keras framework. Custom generative architectures are defined in `fontai.prediction.models`. 


## Installation

clone the repository and install the library by doing `pip install src/fontai`

## Usage

Open source Google fonts can be used for a test run; the main zip file includes around 1k fonts. Just run:

```py
python ./scripts/run-pipeline.py --fit --config-file config/parameters/pipelines/test-run.yaml
```

This will store the processed images in `data/preprocessed-chars` and the trained classifier at `models/supervised-uppercase`. 

Single stages can be executed using the script `scripts/run-single-stage.py`. To see the configuration schema for each stage look at their docstrings, for example:

```py
from fontai.pipeline.stages import FontIngestion, LabeledExampleExtractor, Predictor

help(LabeledExampleExtractor)
```

Finally, the library's API documentation can be accessed [here](https://nestorsag.github.io/textfont-ai/).
