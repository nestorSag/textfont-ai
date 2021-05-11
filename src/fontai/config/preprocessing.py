from pathlib import Path
import logging
import typing as t
import inspect

from pydantic import BaseModel
import strictyaml as yml

import fontai.ingestion.scrappers as scrappers

logger = logging.getLogger(__name__)

CONFIG_SCHEMA = yml.Map({
  "output_folder": yml.Str(), 
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

class Config(BaseModel):
  """
  Wrapper class for the configuration of the ImageExtractor class

  output_folder: folder in which scrapped and zipped ttf/otf files will be saved

  max_zip_size: maximum pre-compression size of zipped output files

  scrappers: list of FileScrapper instances from which scrapped files will be processed.

  """
  output_folder: Path
  max_zip_size: float
  scrappers: t.List[scrappers.FileScrapper]
  yaml: yml.YAML

  # internal BaseModel configuration class
  class Config:
    arbitrary_types_allowed = True

class ConfigHandler(object):
  """
  Wrapper for ingestion's configuration processing logic.

  """

  @classmethod
  def from_string(cls, config: str) -> Config:
    """
    Processes a YAML file and maps it to an Config instance

    file: contents of YAML configuration file

    """

    conf_yaml = yml.load(config, CONFIG_SCHEMA)
    return ConfigHandler.instantiate_config(conf_yaml)

  @classmethod
  def from_file(cls, config: Path) -> Config:
    """
    Processes a YAML file and maps it to an Config instance

    file: Path object pointing to configuration YAML file

    """

    conf_yaml = yml.load(config.read_text(), CONFIG_SCHEMA)
    return ConfigHandler.instantiate_config(conf_yaml)

  @classmethod
  def instantiate_config(cls, config: yml.YAML) -> Config:
    """
    Processes a YAML instance to produce an Config instance.

    config: YAML object from the strictyaml library

    """
    output_folder, max_zip_size, sources = Path(config.data["output_folder"]), config.data["max_zip_size"], config.data["scrappers"]

    #valid_folder = output_folder.exists()
    valid_size = max_zip_size > 0

    # if not valid_folder:
    #   raise Exception(f"output_folder ({output_folder}) does not exist.")
    if not valid_size:
      raise Exception(f"max_zip_size parameter value ({max_zip_size}) is invalid.")

    Path(output_folder).mkdir(parents=True, exist_ok=True)

    source_list = []
    for source in sources:
      kwargs = {} if "kwargs" not in source else source["kwargs"]
      try:
        source_list.append(getattr(scrappers,source["class"])(**kwargs))
      except Exception as e:
        logger.exception(f"Error instantiating FileScrapper object of type {source['class']}")
    return Config(
      output_folder = output_folder, 
      max_zip_size = max_zip_size,
      scrappers = source_list,
      yaml = config)
