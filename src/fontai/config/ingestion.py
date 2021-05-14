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
  def __init__(self):
    
    self.CONFIG_SCHEMA = yml.Map({
      "output_path": yml.Str(), 
      "max_zip_size": yml.Float(), 
      "scrappers": yml.Seq( #scrappers: list of dictionaries with 2 keys: class name and kwargs to be passed
        yml.Map({
          "class":yml.Str(),
          yml.Optional("kwargs"):yml.MapPattern(
            yml.Str(),
            yml.Int() | yml.Float() | yml.Str() | yml.Bool())
        })
      )
    })

  def instantiate_config(self, config: yml.YAML) -> Config:
    """
    Processes a YAML instance to produce an Config instance.

    config: YAML object from the strictyaml library

    """
    output_path, max_zip_size, sources = DataPath(config.data["output_path"]), config.data["max_zip_size"], config.data["scrappers"]

    #valid_folder = output_path.exists()
    valid_size = max_zip_size > 0

    # if not valid_folder:
    #   raise Exception(f"output_path ({output_path}) does not exist.")
    if not valid_size:
      raise ValueError(f"max_zip_size parameter value ({max_zip_size}) is invalid.")

    source_list = []
    for source in sources:
      kwargs = {} if "kwargs" not in source else source["kwargs"]
      try:
        source_list.append(getattr(scrappers,source["class"])(**kwargs))
      except Exception as e:
        logger.exception(f"Error instantiating FileScrapper object of type {source['class']}")
    return Config(
      output_path = output_path, 
      max_zip_size = max_zip_size,
      scrappers = source_list,
      yaml = config)
