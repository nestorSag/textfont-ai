from pathlib import Path
import logging
import typing as t
import inspect

from pydantic import PositiveFloat
import strictyaml as yml

import fontai.ingestion.scrappers as scrappers

from fontai.core.base import BaseConfigHandler, BaseConfig
from fontai.ingestion.scrappers import FreeFontsFileScrapper, DafontsFileScrapper, MultiSourceFileScrapper

logger = logging.getLogger(__name__)

class Config(BaseConfig):
  """
  Wrapper class for the configuration of the IngestionStage class

  max_zip_size: maximum pre-compression size in MB of archives containing font files

  """
  max_zip_size: PositiveFloat


class ConfigHandler(BaseConfigHandler):
  """
  Wrapper for ingestion's configuration processing logic.

  """
  
  def get_config_schema(self):
    
    schema = yml.Map({
      yml.Optional("writer_params", default = {}): self.IO_CONFIG_SCHEMA, 
      yml.Optional("reader_params", default = {}): self.IO_CONFIG_SCHEMA,
      "max_zip_size": yml.Float()
    })

    return schema

  def instantiate_config(self, config: yml.YAML) -> Config:
    """
    Processes a YAML instance to produce an Config instance.

    config: YAML object from the strictyaml library

    """
    output_path, input_path= self.instantiate_io_handler(config.get("output_path")), self.instantiate_io_handler(config.get("input_path"))

    max_zip_size = config.get("max_zip_size").text

    return Config(
      output_path = output_path, 
      input_path = input_path,
      max_zip_size = max_zip_size,
      yaml = config)
