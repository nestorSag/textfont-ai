"""This package contains the codebase of an end-to-end ML pipeline to train generative models for text font generation from open source font files scrapped from the internet. All of the ML stages, from ingestion to model training are designed to be run from configuration files and can be concatenated through a Pipeline class; correct configuration semantics for each one are enforced by the `config` submodule. A brief overview of the submodules is below:

    `config`: Parsers that map YAML configuration files to object wrappers for all the dependencies that the corresponding ML stage needs to be executed either in batch or streaming mode; it makes heavy use of the `strictyaml` and `pydantic` libraries.

    `io`: Contains logic pertaining to data interfaces between ML stages and storage and to other ML stages; this comprises supported Tensorflow record formats, file formats, storage media and file reader and writer classes.

    `runners`: High-level ML stage executors are here, as well as the basic interfaces that each one has to implement.

    `prediction`: Contains the logic for preprocessing Tensorflow examples and get them ready for model training or scoring; also defines custom model architectures and callback functions for training.

    `preprocessing`: Contains the logic for mapping zipped font files to labeled character images ready to be ingested for model training.
  
"""
