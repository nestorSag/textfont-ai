import logging

import typing as t
from pydantic import BaseModel
#from strictyaml import load, Map, Str, Int, Seq, YAMLError
import strictyaml as yml

import fontai.ingestion.retrievers as retrievers

logger = Logging.getLogger(__name__)

class IngestionConfig(BaseModel):
  """
  Wrapper class for the configuration of the Ingestor class

  output_folder: folder in which scrapped and zipped ttf/otf files will be saved

  max_zip_size: maximum pre-compression size of zipped output files

  retrievers: list of StreamRetriever instances from which scrapped files will be processed.

  """
  output_folder: str
  max_zip_size: int
  retrievers: t.List[StreamRetriever]

class ConfigHandler(object):
  """
  Wrapper for ingestion's configuration processing logic.

  """

  self.ingestion_config_schema = strictyaml.Map({
    "output_folder": yml.Str(), 
    "max_zip_size": yml.Int(), 
    "retrievers": yml.Seq(yml.Str())})

  @classmethod
  def parse_ingestion_config_file(file: str) -> IngestionConfig:
    """
    Processes a YAML file and maps it to an IngestionConfig instance

    file: configuration file path

    """

    conf_yaml = strictyaml.load(file, self.ingestion_config_schema)
    return IngestionConfig.instantiate_ingestion_config(conf_yaml):

  @classmethod
  def instantiate_ingestion_config(config: YAML) -> bool:
    """
    Processes a YAML instance to produce an IngestionConfig instance.

    config: YAML object from the strictyaml library

    """
    valid_folder = os.path.isdir(config.output_folder):
    valid_size = config.max_zip_size > 0

    if not valid_folder:
      raise Exception(f"output_folder parameter value ({config.output_folder}) is invalid.")
    if not valid_size:
      raise Exception(f"max_zip_size parameter value ({config.max_zip_size}) is invalid.")

    retriever_list = []
    for retriever_type in config.retrievers:
      if retriever_type in inspect.getmembers(retrievers):
        try:
          retriever_list.append(getattr(retrievers,retriever_type)(**config.retrievers.retriever_type.kwargs))
        except Exception e:
          logger.exception(f"Invalid keyword arguments for retriever of type {retriever_type}")
      else:
        #logger.error(f"Retriever of type {retriever_type} doesn't exist in fontai.ingestion.retrievers")
        raise Exception(f" {retriever_type} is not a subclass of StreamRetriever")

    return IngestionConfig(
      output_folder = config.output_folder, 
      max_zip_size = config.max_zip_size,
      retrievers = retriever_list)
