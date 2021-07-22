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
  max_output_file_size: float

class ConfigHandler(BaseConfigHandler):
  """
  Wrapper for ingestion's configuration processing logic.
  
  """
  @classmethod
  def get_config_schema(self):
    """
    YAML configuration schema:
    
    scrappers: list of subyamls with keys (class, kwargs) specifying the scrapper instances from `fontai.io.scrappers `to use
    output_path: target output folder
    max_output_file_size: Maximum individual output file size in MB

    """
    
    schema = yml.Map({
      "scrappers" : yml.Seq(self.yaml_to_obj.PY_CLASS_INSTANCE_FROM_YAML_SCHEMA),
      yml.Optional("output_path", default = None): self.IO_CONFIG_SCHEMA,
      yml.Optional("max_output_file_size", default = 64.0): yml.Float()
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

    max_output_file_size = config.get("max_output_file_size").data

    return Config(
      max_output_file_size = max_output_file_size,
      output_path = output_path, 
      scrappers = scrappers,
      yaml = config)
