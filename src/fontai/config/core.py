from abc import ABC
import strictyaml as yml
from pathlib import Path

class BaseConfigHandler(ABC):
  """
  Wrapper for each stage's configuration processing logic.

  """

  self.CONFIG_SCHEMA = None

  @classmethod
  def from_string(cls, config: str) -> Config:
    """
    Processes a YAML file and maps it to an Config instance

    file: contents of YAML configuration file

    """

    conf_yaml = yml.load(config, self.CONFIG_SCHEMA)
    return self.instantiate_config(conf_yaml)

  @classmethod
  def from_file(cls, config: Path) -> Config:
    """
    Processes a YAML file and maps it to an Config instance

    file: Path object pointing to configuration YAML file

    """

    conf_yaml = yml.load(config.read_text(), self.CONFIG_SCHEMA)
    return self.instantiate_config(conf_yaml)

  @abstractmethod
  def instantiate_config(self, config: yml.YAML) -> Config:
    """
    Processes a YAML instance to produce an Config instance.

    config: YAML object from the strictyaml library

    """
    pass
