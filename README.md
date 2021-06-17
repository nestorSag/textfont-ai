
# ML pipeline for text font generation

This project addresses all the stages of an ML model whose objective is to train a generative model to come up with new text fonts.

## Organisation

The codebase has 2 main parts, a core python library, `fontai`, in the `src` folder, and a set of scripts built on top of it in the `scripts` folder. The `fontai` package contains the core logic necessary to run the different project's stages, while the `scripts` act as wrappers for `fontai` calls.

## ML projects as code

As this is also an attempt of a production-grade ML pipeline system, execution relies entirely on predefined YAML configuration files. There shoudln't even be a need to star a Python console, instead it is enough to point the corresponding execution script in the `scripts` folder to the correct YAML configuration file. Configuration definitions are flexible enough to enable defining arbitrary Sequential Keras models and changing every parameter in the preprocessing stage, etc. Non-sequential model architectures can be used as well provided they are defined in the `fontai.prediction.models` module.

## Stage summaries

### Ingestion

The data is ingested by web scrapping multiple websites for free font files. This results in 80,400 font zip files from which a bit over 125,000 different fonts can be used.

### Preprocessing

Font files are converted from `ttf` or `otf` to 64x64 `png` images for each alphanumeric character in each font using Apache Beam, and are then batched together into Tensorflow record files for model consumption.

### Training

The models are ran using Tensorflow's Keras framework.
