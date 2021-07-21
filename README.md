
# end-to-end generative ML models for typefaces

This package contains the codebase of an end-to-end ML pipeline to train generative models for text font generation from scrapped open source font files. All of the ML stages, from ingestion to model deployment are designed to be run from a simple entrypoint command passing configuration files with comprehensive schemas. 

## Stages

### Ingestion (urls -> zipped font files)

The data is ingested by scrapping multiple websites for free font files, storing them as zipped `ttf`/`otf` files. At least one of the original scrappers classes (in `fontai.io.scrappers`) have stopped working as of June 2021 due to changes in the structure of the source websites, however the set of Google's fonts can still be reliably downloaded. New scrapper classes for different websites can be easily added.


### Preprocessing (zippedfont files -> Tensorflow record files)

Font files are unzipped and converted from `ttf` or `otf` to k x k `png` images for each alphanumeric character in each font using Apache Beam, and are then batched together into Tensorflow record files for model consumption. They can be stored as individual characters in no particular order or grouping all characters by font.

### Training (Tensorflow record files -> trained model)

The models are ran using Tensorflow's `keras` framework. Custom generative architectures are defined in `fontai.prediction.models`, and are variations of an [adverdarial autoencoder architecture](https://arxiv.org/abs/1511.05644). Any sequential Keras architecture is supported out of the box, and custom architectures defined in said module are also supported.

### Deployment (model -> interactive Dash app)

Trained generative models can be deployed to a small Dash app in which the embedded typeface style space can be explored visually. 

## Setup

If you have a working installation of Tensorflow, cuDNN and CUDA, just clone the repo and install the `fontai` package:

```
git clone https://github.com/nestorSag/textfont-ai.git
cd textfont-ai
pip install -e src/fontai/
```

Otherwise, do `conda env create -f env/conda-env.yaml` right before installing the package to set up the environment.


## Usage

The main entry point to the package is the command  `fontairun` installed with the package. This runs individual stages (i.e, ingestion, preprocessing, training/scoring or deployment) given a valid configuration file for the corresponding stage. To see the few other parameters, and full schema specifications for the different stages, do:

```
fontairun --help
```

Schemas for a small pipeline run that trains a generative model with Google's public fonts with around 4k examples can be found in `config/parameters/google-fonts-examples/`, and a script putting all stages together can be found in `scritps/google-fonts-pipeline.sh` (this can be slow without a GPU). In order to scrape other data sources, new `Scrapper` classes need to be implemented in the `fontai.io.scrappers` module and referenced in the passed configuration file, otherwise font zip files can be downloaded by other means and their location passed to the preprocessing and training stages. The full package's documentation can be found [here](https://nestorsag.github.io/textfont-ai/).

## Experiment tracking

All details of any model-fitting experiment are automatically logged to the configured MLFlow's artifact and backend stores, which default to `./mlruns` if MLFLOW_TRACKING_URI is not set. Those executed from an interactive sessions are also logged, if they're run using the `fontai.runners.Scoring` class. Starting MLFlow's UI with `mlflow ui` allows to explore previous runs.


## Scalability

The preprocessing stages takes a YAML list of command line arguments for Apache Beam, which allows local parallel execution, remote execution using GCP's Dataflow, or any other supported backend. A docker image for remote training can be built by running 

```
docker build -t <username>/fontai-runner:1.0.0 .
```

from the root folder. This image can be deployed to a cloud instance for training, passing the same parameters as for `fontairun`, which is its entry point. In this case, the configuration file must be stored in a reachable location and its full path provided as the `config-file` argument; at the moment, only Google Storage is supported, i.e. a `gs://...` path (Google Storage paths are also supported for input, output and model paths in the configuration file). Other cloud providers can be implemented by extending the `fontai.io.storage` module.

## Quickstart

To run and end-to-end generative pipeline using Google's public fonts, run `scripts/google-fonts-pipeline.sh`. It might take a couple hours.

