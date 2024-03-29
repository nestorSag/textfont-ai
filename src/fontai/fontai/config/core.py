import typing as t
from abc import ABC, abstractmethod
from pathlib import Path
import logging

import strictyaml as yml
from pydantic import BaseModel

from fontai.io.storage import BytestreamPath

logger = logging.getLogger(__name__)


class BasePipelineTransformConfig(BaseModel):

  """
  Base class for ML pipelane stage configuration objects
  
  Args:
      input_path (str, optional): object to list and retrieve input files to be processed
      output_path (str, optional): object to persist output bojects
      yaml (yml.YAML): parsed YAML from supplied configuration file
  
  """
  input_path: t.Optional[str] = None
  output_path: t.Optional[str] = None
  yaml: yml.YAML

  class Config:
    arbitrary_types_allowed = True
  
class SimpleClassInstantiator(object):
  """
  Wrapper for some useful schema definitions and simple class instantiations.
  """

  ANY_PRIMITIVES = yml.Int() | yml.Float() | yml.Bool() | yml.Str() | yml.Seq(yml.Int())

  PY_CLASS_INSTANCE_FROM_YAML_SCHEMA = yml.Map(
        {"class": yml.Str(), 
        yml.Optional("kwargs", default = {}): yml.MapPattern(
          yml.Str(),
          ANY_PRIMITIVES) | yml.EmptyDict()})

  def get_instance(self, yaml: yml.YAML, scope) -> object:
    """
    This method instantiates a class in the global namespace using a string as class name and a dictionary as keyword arguments. This method only works for classes that receive primitive value types as arguments for their constructors.
    
    Args:
        yaml (yml.YAML): AML object that matches the schema given by the PY_CLASS_INSTANCE_FROM_YAML_SCHEMA attribute
        scope (module): module namespace from which the object is to be instantiated.
    
    Returns:
        object: instantiated object
    """
    yaml.revalidate(self.PY_CLASS_INSTANCE_FROM_YAML_SCHEMA)
    return getattr(scope, yaml.get("class").text)(**yaml.get("kwargs").data)
    # try:
    #   yaml.revalidate(self.PY_CLASS_INSTANCE_FROM_YAML_SCHEMA)
    #   return getattr(scope, yaml.get("class").text)(**yaml.get("kwargs").data)
    # except Exception as e:
    #   #print(f"Cannot instantiate class {yaml.get('class').text} from namespace {scope}: {e}")
    #   logger.exception(f"Cannot instantiate class {yaml.get('class').text} from namespace {scope}: {e}")



class BaseConfigHandler(ABC):
  """
  Interface for creating execution configuration objects for ML pipeline stages
  
  Attributes:
      IO_CONFIG_SCHEMA (yml.validators.Validator): schema for I/O parameters
      yaml_to_obj (SimpleClassInstantiator): Helper class to instantiate some Python objects from a YAML definition
  
  """
  yaml_to_obj = SimpleClassInstantiator()

  IO_CONFIG_SCHEMA = yml.Str() | yml.EmptyNone()

  def __init__(self):

    self.other_setup()

    self.CONFIG_SCHEMA = self.get_config_schema()


  def from_string(self, config: str) -> BasePipelineTransformConfig:
    """
    Processes a YAML file and maps it to an Config instance
        
    Args:
        config (str): contents of YAML configuration file
    
    Returns:
        BasePipelineTransformConfig: Configuration object
    
    """

    conf_yaml = yml.load(config, self.CONFIG_SCHEMA)
    return self.instantiate_config(conf_yaml)

  def from_file(self, path: str) -> BasePipelineTransformConfig:
    """
    Processes a YAML file and maps it to an Config instance
        
    Args:
        path (str): Path object pointing to configuration YAML file
    
    Returns:
        BasePipelineTransformConfig: Configuration object
    
    """

    return self.from_string(BytestreamPath(path).read_bytes().decode("utf-8"))

  @abstractmethod
  def instantiate_config(self, config: yml.YAML) -> BasePipelineTransformConfig:
    """
    Processes a YAML instance to produce an Config instance.
        
    Args:
        config (yml.YAML): YAML object from the strictyaml library
    
    """
    pass

  @classmethod
  def get_config_schema(self):

    return None

  def other_setup(self):
    pass
