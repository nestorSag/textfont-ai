"""This module contains the base interfaces implemented by ML pipeline stages and pipeline objects.

"""
from __future__ import annotations
import logging
import typing
import sys
import typing as t
from abc import ABC, abstractmethod


from fontai.config.core import BasePipelineTransformConfig
from fontai.io.formats import InMemoryFile
from fontai.io.readers import ReaderClassFactory
from fontai.io.writers import WriterClassFactory

logger = logging.Logger(__name__)
  
class Transform(ABC):

  """This class is the primary interface implemented by any ML processing stage; it has a process method for real-time processing, and a transform_batch method to process a set of files and persist the results back to storage.
  """

  input_file_format = InMemoryFile
  output_file_format = InMemoryFile

  @property
  def reader_class(self):
    return ReaderClassFactory.get(self.input_file_format)

  @property
  def writer_class(self):
    return WriterClassFactory.get(self.output_file_format)
  
  def transform(self, data: t.Any) -> t.Any:
    """Processes a single data instance.
    
    Args:
        data (t.Any): Input data
    """
    pass

  # @classmethod
  # @abstractmethod
  # def transform_batch(self, input_path: str, output_path: str) -> None:
  #   """Processes a batch of files and persist the results back to storage.
    
  #   Args:
  #       input_path (str): Input folder
  #       output_path (str): Output folder
  #   """
  #   pass

class IdentityTransform(Transform):

  """This class applies an identity transformation to its inputs; it is useful for ML stages that are only active in thetraining stage and not on the deployment stage.
  """

  def transform(self, data: t.Any):

    return data

  # def transform_batch(self, input_path: str, output_path: str):
  #   writer = writer(output_path)
  #   for file in reader(input_path).get_files():
  #     try:
  #       file = file.deserialise()
  #       try:
  #         writer.write(self.process(file))
  #       except Exception as e:
  #         logger.info(f"Error writing file: {e}")
  #     except Exception as e:
  #       logger.exception(f"Error reading file: {e}")


class ConfigurableTransform(Transform):

  """Interface for configurable tranformations; they can be instantiated and run from YAML configuration files.
  """

  @classmethod
  @abstractmethod
  def from_config_object(cls, config: BasePipelineTransformConfig) -> ConfigurableTransform:
    """Instantiate class from a configuration object
    
    Args:
        config (BasePipelineTransformConfig): Config object parsed from a YAML file
    """
    pass

  @classmethod
  def from_config_file(cls, path: str, **kwargs) -> ConfigurableTransform:
    """Create a ConfigurableTransform instance from a YAML configuration file
    
    Args:
        path (str): Path to the YAML configuration file
    
    Returns:
        ConfigurableTransform: Instance created from configuration file.
    """
    return cls.from_config_object(cls.parse_config_file(path), **kwargs)

  @classmethod
  def from_config_str(cls, yaml: str) -> ConfigurableTransform:
    """Create a ConfigurableTransform instance from the content of a YAML configuration file
    
    Args:
        yaml (str): YAML file content
    
    Returns:
        ConfigurableTransform: Instance created from configuration file.
    """
    return cls.from_config_object(cls.parse_config_str(yaml))

  @classmethod
  def parse_config_file(cls, path: str) -> BasePipelineTransformConfig:
    """Parse a YAML configuration file and create an instance inheriting from BasePipelineTransformConfig
    
    Args:
        path (str): path to YAML configuration file
    
    Returns:
        BasePipelineTransformConfig: Instantiated Config instance
    """
    return self.get_config_parser().from_file(path)

  @classmethod
  def parse_config_str(cls, config_str: str) -> BasePipelineTransformConfig:
    """Parse the contents of a YAML configuration file and create an instance inheriting from BasePipelineTransformConfig
    
    Args:
        config_str (str): YAML content in string format
    
    Returns:
        BasePipelineTransformConfig: Instantiated Config instance
    """
    return cls.get_config_parser().from_string(config_str)


  @classmethod
  def run_from_config_file(cls, path: str) -> None:

    """Instantiate and run transform oject from a YAML file
    
    Args:
        path (str): Path to the YAML file
    """
    
    config = cls.parse_config_file(path)
    cls.run_from_config(config)

  @classmethod
  @abstractmethod
  def run_from_config_object(cls, config: BasePipelineTransformConfig) -> None:
    """Instantiate and run transform from configuration object
    
    Args:
        config (BasePipelineTransformConfig): Configuration object.
    """
    pass
    #cls.from_config_file(path).run_from_config()


  @classmethod
  @abstractmethod
  def get_config_parser(cls) -> BasePipelineTransformConfigHandler:

    """
    Returns an instance of the transform's configuration parser class
    
    """
    pass


class FittableTransform(ConfigurableTransform):

  """Interface for pipeline transforms that can be fitted. Scoring is done using the 'transform' method.
  """

  def fit(self, data: t.Any) -> FittableTransform:
    """
    Fits the stage to the passed data
    
    """

    pass

  @classmethod
  def fit_from_config_file(cls, path: str) -> FittableTransform:
    """
    Fits the transform's model according to a YAML configuration file
    
    Args:
        path (str): Path to YAML file
    
    """
    config = cls.parse_config_file(path)
    return cls.fit_from_config_object(config)

  @classmethod
  @abstractmethod
  def fit_from_config_object(self, config: BasePipelineTransformConfig) -> FittableTransform:
    """
    Fits the transform's model according to a configuration object
    
    Args:
        config (BasePipelineTransformConfig): Configuration object.
    
    """
    pass
