import typing as t
from abc import ABC, abstractmethod
import strictyaml as yml
from pathlib import Path

from pydantic import BaseModel

class BaseConfigHandler(ABC):
  """
  Interface for creating ML pipeline stages' execution configuration objects

  """
  def __init__(self):

    self.ANY_PRIMITIVES = yml.Int() | yml.Float() | yml.Str() | yml.Bool()

    self.PY_CLASS_INSTANCE_FROM_YAML_SCHEMA = yml.Map(
          {"class": yml.Str(), 
          Optional("kwargs"): yml.MapPattern(
            yml.Str(),
            self.ANY_PRIMITIVES)})

    self.CONFIG_SCHEMA: t.Optional[yml.Map] = None

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
