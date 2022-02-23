
# end-to-end generative ML models for typefaces

This package contains the codebase of an end-to-end ML pipeline to train generative models for text font generation from scrapped open source font files. All of the ML stages, from ingestion to model deployment are designed to be run from a simple entrypoint command passing configuration files with comprehensive schemas. 

<p align="center" style="font-size:20px; margin:10px 10px 0px 10px">
    <em>Generative font model example</em>
</p>
<p align="center" style="font-size:20px; margin:10px 10px 40px 10px">
  <img src="https://www.nestorsag.com/assets/static/tensor-fonts.a0d7643.332718c7c7b045b834cafc9cedd93a5f.gif" alt="Generative font example" width="640px">
</p>


- [Setup](#setup)
  - [Usage](#usage)
  - [Docs](#docs)
  - [Test runs](#test-runs)
- [Lifecycle Stages](#stages)
  - [Ingestion](#ingestion)
  - [Preprocessing](#preprocessing)
  - [Training](#training)
  - [Deployment](#deployment)
- [Features](#features)
  - [Experiment tracking](#experiment-tracking)
  - [Scalability](#scalability)
  - [Supported model architectures](#supported-model-architectures)
  - [Input preprocessing](#4input-preprocessing)


## Setup

* Clone the repo

* If there is a working installation of Tensorflow, cuDNN and CUDA, just install the `fontai` package from the root folder:

```
pip install -e src/fontai/
```

* Otherwise, you can setup the environment with Miniconda, also from the root folder:

```
conda env create -f conda-env.yaml
conda activate textfont-ai
pip install -e src/fontai/
```

### Usage

The main entry point to the package is the command  `fontairun` installed with the package. This runs individual stages (i.e, ingestion, preprocessing, training/scoring or deployment) given a valid configuration file for the corresponding stage. To see other supported parameters, and full config schema specifications for different stages, do:

```
fontairun --help
```

### Docs

[API docs](https://nestorsag.github.io/textfont-ai/)

### Test runs

Schemas for a small pipeline run that trains a generative model with Google's public fonts with around 4k examples can be found in `config/parameters/google-fonts-examples/`, and a script putting all stages together can be found in `scripts/google-fonts-pipeline.sh` (running this can be slow without a GPU). In order to scrape other data sources, new `Scrapper` classes need to be implemented in the `fontai.io.scrappers` module and referenced in the passed configuration file, otherwise font zip files can be downloaded by other means and their location passed to the preprocessing and training stages.

To run and end-to-end generative pipeline locally using Google's public fonts, run `scripts/google-fonts-pipeline.sh`. It might take a couple hours.

Pretrained decoder models for character and font styles are in the `pretrained` folder, and deployment configurations are in `config/parameters/deploy-pretrained`. To deploy a Dash web app to visualise the style space from the char model, do:

```
fontairun --stage deployment --config-file config/parameters/deploy-pretrained/chars-style-model.yaml
```

## Stages

### Ingestion

The data is ingested by scrapping multiple websites for free font files, storing them as zipped `ttf`/`otf` files. At least one of the original scrappers classes (in `fontai.io.scrappers`) have stopped working as of June 2021 due to changes in the structure of the source websites, however the set of Google's fonts can still be reliably downloaded. New scrapper classes for different websites can be easily added.


### Preprocessing 

Font files are unzipped and converted from `ttf` or `otf` to `k x k` PNG images for each alphanumeric character in each font using Apache Beam, and are then batched together into Tensorflow record files for model consumption. They can be stored as individual characters in no particular order or grouping all characters by font. The latter is useful to train some specific generative architectures in the `fontai.scoring.models` module.

### Training 

The models are ran using Tensorflow's `keras` framework. Custom generative architectures are defined in `fontai.prediction.models`, and are variations of an [adversarial autoencoder architecture](https://arxiv.org/abs/1511.05644). Any sequential Keras architecture is supported out of the box, and custom architectures defined in said module are also supported.

### Deployment 

Trained generative models can be deployed to a small Dash app in which the embedded typeface style space can be explored interactively. 


## Features

### Experiment tracking

All details of any model-fitting experiment are automatically logged to the configured MLFlow's artifact and backend stores, which default to `./mlruns` if MLFLOW_TRACKING_URI is not set. Those executed from an interactive sessions are also logged, if they're run using the `fontai.runners.Scoring` class. Starting MLFlow's UI with `mlflow ui` allows to explore previous runs.


### Scalability

The preprocessing stages takes a YAML list of command line arguments for Apache Beam, which allows local parallel execution, remote execution using GCP's Dataflow, or any other Beam-supported backend. A docker image for remote training can be built by running 

```
docker build -t <username>/fontai-runner:1.0.0 .
```

For the particular case of GCP AI platform, it's better to use a different build; look at the comments in the Dockerfile. This image can be deployed to a cloud instance for training, passing the same parameters as for the `fontairun` command, which is its entry point. In this case, the configuration file must be stored in a reachable location and its full path provided as the `config-file` argument; at the moment, only Google Storage is supported, i.e. a `gs://...` path (Google Storage paths are also supported for data input, output and model paths in the configuration file). Other cloud providers can be implemented by extending the `fontai.io.storage` module.


### Supported model architectures

Sequential models from Keras are supported out of the box by specifying them on configuration files; custom generative architectures are defined in the `fontai.prediction.models` module. There are two kind of custom generative architectures:

* Character style [`CharStyleSAAE` class]: These models take batches of individual, randomly chosen characters from different fonts, and use an adversarial autoencoder architecture to embed character styles in a latent space. 

* Font style [`PureFontStyleSAAE` class, `TensorFontStyleSAAE` class]: These architectures take batches of entire fonts and try to use information from all characters to encode the font's style. 

The main difference is that for font style models, a given style vector will map to consistent character styles across the font's character set, which is not necessarily true for character style models.

<p align="center" style="font-size:20px; margin:10px 10px 0px 10px">
    <em>Generative font model example</em>
</p>
<p align="center" style="font-size:20px; margin:10px 10px 40px 10px">
  <img src="https://www.nestorsag.com/assets/static/architecture.cbab2cf.e7156abd33fd01b1a301f1e27a886678.png" alt="One of the custom model architectures" width="640px">
</p>

For more details, see [this](https://www.nestorsag.com/blog/font2vec-generative-models-for-typefaces/) informal blog post.

### Input preprocessing

Model inputs can be mapped and filtered with custom transformations defined in `fontai.prediction.custom_filters` and `fontai.prediction.custom_mappers` before being used for training, for example to just use fonts with certain name patterns such as 'serif' or '3d', or to round normalised pixels to 0 or 1; another good example is that since free fonts usually come with malformed characters or symbol fonts that don't look like characters, using batches of scored images that have been passed through a classifier can be useful to filter misclassified or low-confidence images. This generally improves the final generative model's quality. For such a filter, input records must have the `ScoredLabeledChar` or `ScoredLabeledFont` schema, in the `fontai.io.records` module.
