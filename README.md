
# Generative models for text font generation

This project addresses all the stages of an ML model whose objective is to train a generative model to come up with new text fonts, provided different types of user-supplied constraints.

## Organisation

The codebase has 2 main parts, a core python library, `fontai`, in the `src` folder, and a set of scripts built on top of it in the `scripts` folder. The `fontai` package contains the core logic necessary to run the different project's stages((such as web scrapping functions and preprocessing logic particular to the project's data formats), while the `scripts` act as wrappers for `fontai` calls that add parameters such as data sources, sinks ,execution engines, model hyperparameters and so on.

## Stage summaries

### Ingestion

The data is ingested by web scrapping multiple websites for free font files. This results in 80,400 font zip files from which a bit over 125,000 different fonts can be used.

### Preprocessing

Font files are converted from `ttf` or `otf` to 64x64 `png` images for each alphanumeric character in each font using Apache beam with Google's Dataflow engine, and are stored in Cloud Storage as TFRecord files.

### Training

TensorFlow is used to train the models; training calls are done by passing a comprehensive hyperparameter JSON file to a Docker container with suitable permissions that runs the optimisation.

