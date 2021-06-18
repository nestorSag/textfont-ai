from pathlib import Path
import logging
import typing as t
import inspect

from pydantic import PositiveFloat
import strictyaml as yml

import fontai.io.scrappers as scrapper_module

from fontai.config.core import BaseConfigHandler, BasePipelineTransformConfig

logger = logging.getLogger(__name__)


class Config(BasePipelineTransformConfig):

  scrappers: t.List[scrapper_module.Scrapper]

class ConfigHandler(BaseConfigHandler):
  """
  Wrapper for ingestion's configuration processing logic.
  
  """
  
  def get_config_schema(self):
    
    schema = yml.Map({
      "scrappers" : yml.Seq(self.yaml_to_obj.PY_CLASS_INSTANCE_FROM_YAML_SCHEMA),
      yml.Optional("output_path", default = None): self.IO_CONFIG_SCHEMA
    })

    return schema

  def instantiate_config(self, config: yml.YAML) -> Config:
    """
    Processes a YAML instance to produce an Config instance.
        
    Args:
        config (yml.YAML): YAML object from the strictyaml library
    
    Returns:
        Config: Configuration object
    
    """
    output_path = config.get("output_path").text

    scrappers = [self.yaml_to_obj.get_instance(yaml=scrapper, scope=scrapper_module) for scrapper in config.get("scrappers")]

    return Config(
      output_path = output_path, 
      scrappers = scrappers,
      yaml = config)
