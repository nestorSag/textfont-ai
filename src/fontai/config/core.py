import typing as t
from abc import ABC, abstractmethod
from pathlib import Path
import logging

import strictyaml as yml
from pydantic import BaseModel

from fontai.core.io import BytestreamPath

logger = logging.getLogger(__name__)

class IngestionConfig(BaseModel):

  """
    Base class for ingestion stage's configuration objects

    scrappers: object to list and retrieve input files to be processed

    output_path: object to persist output bojects

    yaml: parsed YAML from supplied configuration file

  """
  scrappers: t.Optional[Scrapper]
  output_path: t.Optional[str]
  yaml: yml.YAML

  class Config:
    arbitrary_types_allowed = True


class BasePipelineTransformConfig(BaseModel):

  """
    Base class for ML pipelane stage configuration objects

    input_path: object to list and retrieve input files to be processed

    output_path: object to persist output bojects

    yaml: parsed YAML from supplied configuration file

  """
  input_path: t.Optional[str] = None
  output_path: t.Optional[str] = None
  reader: t.Optional[BatchReader] = None
  writer: t.Optional[BatchWriter] = None
  yaml: yml.YAML

  class Config:
    arbitrary_types_allowed = True
  
class SimpleClassInstantiator(object):
  """
    Wrapper for some useful schema definitions and simple class instantiations.
  """
  def __init__(self):

    self.ANY_PRIMITIVES = yml.Int() | yml.Float() | yml.Str() | yml.Bool()

    self.PY_CLASS_INSTANCE_FROM_YAML_SCHEMA = yml.Map(
          {"class": yml.Str(), 
          yml.Optional("kwargs", default = {}): yml.MapPattern(
            yml.Str(),
            self.ANY_PRIMITIVES) | yml.EmptyDict()})

  def get_instance(csl, yaml: yml.YAML) -> object:
    """
      This method instantiates a class in the global namespace using a string as class name and a dictionary as keyword arguments. This method only works for classes that receive primitive value types as arguments for their constructors.

      yaml: YAML object that matches the schema given by the PY_CLASS_INSTANCE_FROM_YAML_SCHEMA attribute
    """
    try:
      yaml.revalidate(self.PY_CLASS_INSTANCE_FROM_YAML_SCHEMA)
      return globals()[yaml.get("class").text](**yaml.get("kwargs").data)
    except Exception as e:
      logger.exception(f"Cannot instantiate class {yaml.get("class").text} from global namespace: {e}")



class BaseConfigHandler(ABC):
  """
  Interface for creating execution configuration objects for ML pipeline stages

  """
  def __init__(self):

    self.yaml_to_obj = SimpleClassInstantiator()

    self.IO_CONFIG_SCHEMA = yml.Str() | yml.EmptyNone()

    self.other_setup()

    self.CONFIG_SCHEMA: t.Optional[yml.Map] = self.get_config_schema()


  def instantiate_io_handlers(self, yaml: yml.YAML):
    if yaml.get("input_path").data is not None and yaml.get("reader").data is not None:
      reader = globals()[yaml.get("reader").data](input_path = yaml.get("input_path").data)
    else:
      reader = None

    if yaml.get("output_path").data is not None and yaml.get("writer").data is not None:
      writer = globals()[yaml.get("writer").data](output_path = yaml.get("output_path").data)
    else:
      writer = None

    return reader, writer

  def from_string(self, config: str) -> BaseModel:
    """
    Processes a YAML file and maps it to an Config instance

    file: contents of YAML configuration file

    """

    conf_yaml = yml.load(config, self.CONFIG_SCHEMA)
    return self.instantiate_config(conf_yaml)

  def from_file(self, config: BytestreamPath) -> BaseModel:
    """
    Processes a YAML file and maps it to an Config instance

    file: Path object pointing to configuration YAML file

    """

    conf_yaml = self.from_string(config.read_bytes().decode("utf-8"))
    return self.instantiate_config(conf_yaml)

  @abstractmethod
  def instantiate_config(self, config: yml.YAML) -> BaseModel:
    """
    Processes a YAML instance to produce an Config instance.

    config: YAML object from the strictyaml library

    """
    pass

  def get_config_schema(self):

    return None

  def setup(self):
    pass