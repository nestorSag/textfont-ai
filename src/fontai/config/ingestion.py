from pathlib import Path
import logging
import typing as t
import inspect

from pydantic import PositiveFloat
import strictyaml as yml

from fontai.ingestion.scrappers import *

from fontai.core.base import IngestionConfig, BaseConfigHandler

logger = logging.getLogger(__name__)


class ConfigHandler(BaseConfigHandler):
  """
  Wrapper for ingestion's configuration processing logic.

  """
  
  def get_config_schema(self):
    
    schema = yml.Map({
      yml.Optional("scrappers"): yml.Seq(self.yaml_to_obj.PY_CLASS_INSTANCE_FROM_YAML_SCHEMA) 
      yml.Optional("output_path", default = None): self.IO_CONFIG_SCHEMA
    })

    return schema

  def instantiate_config(self, config: yml.YAML) -> Config:
    """
    Processes a YAML instance to produce an Config instance.

    config: YAML object from the strictyaml library

    """
    output_path = self.instantiate_io_handler(config.get("output_path"))

    scrappers = [self.yaml_to_obj.get_instance(scrapper) for scrapper in config.get("scrappers")]

    return IngestionConfig(
      output_path = output_path, 
      scrappers = scrappers,
      yaml = config)
