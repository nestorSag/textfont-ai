from pathlib import Path
import logging
import typing as t
import inspect

from pydantic import BaseModel, PositiveFloat
import strictyaml as yml

import fontai.ingestion.scrappers as scrappers

from fontai.core import DataPath
from fontai.config.core import BaseConfigHandler

logger = logging.getLogger(__name__)

class Config(BaseModel):
  """
  Wrapper class for the configuration of the Ingestor class

  output_path: DataPath instance representing the folder in which scrapped and zipped ttf/otf files will be saved

  max_zip_size: maximum pre-compression size of zipped output files

  scrappers: list of FileScrapper instances from which scrapped files will be processed.

  """
  output_path: DataPath
  max_zip_size: PositiveFloat
  scrappers: t.List[scrappers.FileScrapper]
  yaml: yml.YAML

  # internal BaseModel configuration class
  class Config:
    arbitrary_types_allowed = True



class ConfigHandler(BaseConfigHandler):
  """
  Wrapper for ingestion's configuration processing logic.

  """
  
  def get_config_schema(self):
    
    schema = yml.Map({
      "output_path": yml.Str(), 
      "max_zip_size": yml.Float(), 
      "scrappers": yml.Seq(self.PY_CLASS_INSTANCE_FROM_YAML_SCHEMA
      )
    })

    return schema

  def instantiate_config(self, config: yml.YAML) -> Config:
    """
    Processes a YAML instance to produce an Config instance.

    config: YAML object from the strictyaml library

    """
    output_path, max_zip_size, scrappers = DataPath(config.data["output_path"]), config.data["max_zip_size"], config.get("scrappers")

    #valid_folder = output_path.exists()
    valid_size = max_zip_size > 0

    # if not valid_folder:
    #   raise Exception(f"output_path ({output_path}) does not exist.")
    if not valid_size:
      raise ValueError(f"max_zip_size parameter value ({max_zip_size}) is invalid.")

    source_list = []
    for scrapper in scrappers:
      try:
        instance = self.yaml_to_obj.get_instance(scrapper)
        source_list.append(instance)
      except Exception as e:
        logger.exception(f"Error instantiating FileScrapper object of type {source['class']}")

    if len(source_list) == 0:
      raise ValueError("List of scrappers is empty. There are no sources to process.")

    return Config(
      output_path = output_path, 
      max_zip_size = max_zip_size,
      scrappers = source_list,
      yaml = config)
