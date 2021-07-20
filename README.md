
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

If you a working installation of Tensorflow, cuDNN and CUDA, just clone the repo and install the `fontai` package:

```
git clone https://github.com/nestorSag/textfont-ai.git
cd textfont-ai
pip install -e src/fontai/
```

Otherwise, do `conda env create -f env/conda-env.yaml` right before installing the package to set up the environment.


## Usage

The main entry point to the package is the command  `fontairun` installed with the package. This runs individual stages (i.e, ingestion, preprocessing or training/scoring) given a valid configuration file for the corresponding stage. To see the few other parameters, and full schema specifications for the different stages, do:

```
fontairun --help
```

Schemas for a small pipeline run that trains a generative model with Google's public fonts with around 4k examples can be found in `config/parameters/google-fonts/`, and a script putting all stages together can be found in `scritps/google-fonts.sh` (this can be slow without a GPU). In order to scrape other data sources, new `Scrapper` class need to be implemented in the `fontai.io.scrappers` module and referenced in the passed configuration file, otherwise font zip files can be downloaded by other means and their location passed to the preprocessing and training stages. The full package's documentation can be found [here](https://nestorsag.github.io/textfont-ai/).

## Experiment tracking

All details of any model-fitting experiment are automatically logged to the configured MLFlow's artifact and backend stores, which default to `./mlruns`. Those executed from an interactive sessions are also logged, if they're run using the `fontai.runners.Scoring` class.


## Scalability

The preprocessing stages takes a YAML list of command line arguments for Apache Beam, which allows local parallel execution or remote execution using GCP's Dataflow provided valid GCP credentials or any other backend. A docker image for remote training can be built by doing 

```
docker build -t <username>/fontai-runner:1.0.0 -f docker-images/training/Dockerfile .
```

from the root folder. This image can be deployed to a cloud instance for training, passing the same parameters as for `fontairun`. In this case, the configuration file must be stored in a reachable location, and for the moment, only GCP is supported, by providing full paths, i.e. `gs://...` in the image's arguments. Other cloud providers can be implements by extending the `fontai.io.storage` module.
