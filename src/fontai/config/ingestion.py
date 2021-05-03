import logging
from pathlib import Path

import typing as t
from pydantic import BaseModel
#from strictyaml import load, Map, Str, Int, Seq, YAMLError
import strictyaml as yml

import fontai.ingestion.retrievers as retrievers

logger = Logging.getLogger(__name__)

CONFIG_SCHEMA = strictyaml.Map({
  "output_folder": yml.Str(), 
  "max_zip_size": yml.Float(), 
  "retrievers": yml.Seq( #retrievers: list of dictionaries with 2 keys: class name and kwargs to be passed
    yml.Map({
      "class":yml.Str(),
      yml.Optional("kwargs"):yml.MapPattern(
        yml.Str(),
        yml.Int() | yml.Float() | yml.Str() | yml.Bool())
    })
  )
})

class Config(BaseModel):
  """
  Wrapper class for the configuration of the Ingestor class

  output_folder: folder in which scrapped and zipped ttf/otf files will be saved

  max_zip_size: maximum pre-compression size of zipped output files

  retrievers: list of StreamRetriever instances from which scrapped files will be processed.

  """
  output_folder: Path
  max_zip_size: float
  retrievers: t.List[StreamRetriever]

class ConfigHandler(object):
  """
  Wrapper for ingestion's configuration processing logic.

  """

  @classmethod
  def parse_config(config: str) -> Config:
    """
    Processes a YAML file and maps it to an Config instance

    file: configuration file path

    """

    conf_yaml = strictyaml.load(config, CONFIG_SCHEMA)
    return ConfigHandler.instantiate_config(conf_yaml):

  @classmethod
  def instantiate_config(config: YAML) -> Config:
    """
    Processes a YAML instance to produce an Config instance.

    config: YAML object from the strictyaml library

    """
    output_folder, max_zip_size, sources = Path(config.data["output_folder"]), config.data["max_zip_size"], config.data["retrievers"]

    valid_folder = output_folder.exists():
    valid_size = max_zip_size > 0

    if not valid_folder:
      raise Exception(f"output_folder ({output_folder}) does not exist.")
    if not valid_size:
      raise Exception(f"max_zip_size parameter value ({max_zip_size}) is invalid.")

    instance_list = []
    for instance in instances:
      if instance["class"] in inspect.getmembers(retrievers):
        kwargs = {} if "kwargs" not in instance else instance["kwargs"]
        try:
          instance_list.append(getattr(retrievers,instance["class"])(**kwargs))
        except Exception e:
          logger.exception(f"Invalid keyword arguments for retriever of type {instance["class"]}")
      else:
        #logger.error(f"Retriever of type {retriever_type} doesn't exist in fontai.ingestion.retrievers")
        raise Exception(f" {instance["class"]} is not a subclass of StreamRetriever")

    return Config(
      output_folder = output_folder, 
      max_zip_size = max_zip_size,
      retrievers = instance_list)
