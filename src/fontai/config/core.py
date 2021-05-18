import typing as t
from abc import ABC, abstractmethod
from pathlib import Path
import logging

import strictyaml as yml
from pydantic import BaseModel

logger = logging.getLogger(__name__)

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

    self.other_setup()

    self.CONFIG_SCHEMA: t.Optional[yml.Map] = self.get_config_schema()


  def from_string(self, config: str) -> BaseModel:
    """
    Processes a YAML file and maps it to an Config instance

    file: contents of YAML configuration file

    """

    conf_yaml = yml.load(config, self.CONFIG_SCHEMA)
    return self.instantiate_config(conf_yaml)

  def from_file(self, config: Path) -> BaseModel:
    """
    Processes a YAML file and maps it to an Config instance

    file: Path object pointing to configuration YAML file

    """

    conf_yaml = yml.load(config.read_text(), self.CONFIG_SCHEMA)
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
