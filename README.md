
# ML pipeline for text font generation [under development]

This package contains the codebase of an end-to-end MLOps pipeline to train generative models for text font generation from open source font files scrapped from the internet. All of the ML stages, from ingestion to model deployment are designed to be run from simple entrypoint scripts passing configuration files with well defined schemas. 

## Stages

### Ingestion (urls -> zipped font files)

The data is ingested by scrapping multiple websites for free font files, storing them as zipped `ttf`/`otf` files. At least one of the original scrappers classes (in `fontai.io.scrappers`) have stopped working as of June 2021 due to changes in the structure of the source websites, however the set of Google's fonts can still be reliably downloaded. New scrapper classes for different websites can be easily added.


### Preprocessing (zippedfont files -> Tensorflow record files)

Font files are unzipped and converted from `ttf` or `otf` to k x k `png` images for each alphanumeric character in each font using Apache Beam, and are then batched together into Tensorflow record files for model consumption. They can be stored as individual characters in no particular order or grouping all characters by font.

### Training (Tensorflow record files -> trained model)

The models are ran using Tensorflow's `keras` framework. Custom generative architectures are defined in `fontai.prediction.models`, and are variations of an [adverdarial autoencoder architecture]().


## Setup

The project uses `conda` for environment management. Set it up with:

```
git clone https://github.com/nestorSag/textfont-ai.git
cd textfont-ai
conda env create -f environment.yaml
pip install -e src/

```
## Usage

There are 3 entry points scripts

- `scripts/base/run-stage.py`: Run a single stage (i.e. ingestion, preprocessing and training/scoring) by passing an appropriate execution configuration file. Use `--help` for details of the stages' configuration schema. Examples of config files can be found in `config/parameters`

- `scripts/base/run-pipeline.py`: Run a sequence of stages by passing an appropriate pipeline configuration file. use `--help` for details of configuration schema. There can be any number of stages of the same type, e.g. for training a model that uses the scorings for a previous model to filter examples.

- `scripts/base/run-grid-search.py`: Perform grid search on a model's configuration file's parameters using a parameter grid specified through a JSON file. use `--help` for details.


## Experiment tracking

All details of any model-fitting experiment are automatically logged to the configured MLFlow's artifact and backend stores, which default to `./mlruns`. If `run-grid-search.py` is used, they are grouped into experiments. Those executed from an interactive sessions are also logged, if they're run using the `fontai.runners.Scoring` class.


## Scalability

The preprocessing stages takes a YAML list of command line arguments for Apache Beam, which allows local parallel execution or remote execution using GCP's Dataflow, provided valid GCP credentials; by default, GOOGLE_APPLICATION_CREDENTIALS points to `config/credentials/gcp-credentials.json` in the environment, so putting them there would be enough.

In the case of remote training or grid search, and assuming GCP's credentials are present, a Docker image of the environment is uploaded to the AI plattform to be used for training.

## Deployment 

A trained generative model can be deployed to a Docker container of a Dash application, in which the embeded font style space can be explored visually and interactively.


# Quickstart

 A generative model using Google fonts (~1k) can be trained locally by doing

```py
python ./scripts/run-pipeline.py --config-file config/parameters/pipelines/generative-using-google-fonts.yaml
```

Finally, detailed documentation can be found [here](https://nestorsag.github.io/textfont-ai/).

## Notes

Please be aware that some environment variables have been set to modify Tensorflow's default behaviour, including logging and GPU memory allocation.