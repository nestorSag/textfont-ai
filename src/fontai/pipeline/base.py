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
  
  

  @classmethod
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
  @abstractmethod
  def run_from_config_file(cls, path: str) -> None:

    pass
    #cls.from_config_file(path).run_from_config()

  @classmethod
  @abstractmethod
  def run_from_config_object(cls, config: BasePipelineTransformConfig) -> None:

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

  def fit(self) -> FittableTransform:
    """
    Fits the stage to the passed data

    """
    pass

  @classmethod
  @abstractmethod
  def fit_from_config_file(self) -> FittableTransform:
    """
    Fits the stage to the passed data

    """
    pass

  @classmethod
  @abstractmethod
  def fit_from_config(self) -> FittableTransform:
    """
    Fits the stage to the passed data

    """
    pass


  # def load(self, input_folder: str) -> FittableMLPipelineTransform:
  #   loaded = super().load(input_folder)
  #   loaded.model = Model.load(input_folder)
  #   return loaded




# class MLPipelineTransform(ConfigurableTransform, ABC):
#   """
#     Interface implemented by runnable ML stage objects. They are initialised by passing an execution configuration object and can perform batch transforming according to it; they can also do stream transforming using the `transform` method.

#     config: Execution configuration object
#   """


#   @abstractmethod
#   def transform_batch(self, input_path: str, output_path: str) -> None:
#     """
#     transformes a batch of files and persist output
    
#     Args:
#         reader (BatchReader): Object inheriting from BatchReader
#         writer (BatchWriter): Object inheriting from BatchWriter
    
#     """
#     pass


#   @classmethod
#   @abstractmethod
#   def get_stage_name(cls) -> str:
#     """
#     Returns a string with the stage name
    
#     """
#     pass

#   def save(self, output_folder: str) -> None:
#     """
#     Persists necessary data from which the instance can be loaded again.
    
#     output_folder: BytestreamPath instance pointing to output folder.
    
#     Args:
#         output_folder (str): path to output folder in storage media
    
#     """
#     output_folder = BytestreamPath(output_folder)

#     logger.info(f"{self.__class__.__name__} configuration persisted at {output_folder}")
#     (output_folder / "config.yaml").write_bytes(bytes(self.config.yaml.as_yaml().encode("utf-8")))

#   @classmethod
#   def load(cls, input_folder: str) -> MLPipelineTransform:
#     """
#     Load an instance of this class
        
#     Args:
#         input_folder (str): Path to input folder
    
#     Returns:
#         MLPipelineTransform: Loaded instance
    
#     """
#     logger.info(f"{self.__class__.__name__} loaded from {input_folder}")
#     return cls.get_config_parser().from_file(input_folder / "config.yaml")    






# class MLPipeline(MLPipelineTransform):
#   """
#   Wrapper class for a sequence of ML pipeline stages; can both fit stages that are fittable and process both batch and in-memory data
    
#   Attributes:
#       stages (t.List[MLPipelineTransform]): List of pipeline stages.
  
#   """

#   def __init__(self, stages: t.List[MLPipelineTransform]):

#     for stage in stages:
#       if not isinstance(stage, MLPipelineTransform):
#         raise TypeError("Some stages are not instances of MLPipelineTransform")

#     self.stages = stages

#   class StageWrapper(object):
#     """
#     Wrapper class for a single ML stage for one-to-many transformations.

#     """

#     def __init__(stage: MLPipelineStage):
#       self.stage = stage

#     def process(self, data: t.Generator[t.Any, None, None]) -> t.Generator[t.Any, None, None]:
      
#       for elem_in in data:
#         for elem_out in self.stage.process(elem_in):
#           yield elem_out

#     def fit(self, data: t.Generator[t.Any, None, None]) -> t.Generator[t.Any, None, None]:
      
#       if isinstance(self.stage, FittableStage):
#         self.stage.fit(data)

#     def fit_and_process(self, data: t.Generator[t.Any, None, None]) -> t.Generator[t.Any, None, None]:
      
#       self.fit(data)
#       return self.process(data)


#   def run_from_config(self) -> None:

#     self.transform_batch(self.config.input_path, self.config.output_path)

#   def transform(self, input_data: t.Any) -> t.Any:
#     """
#     transformes a single data instance

#     """

#     data_in = (input_data)
#     wrapped_stages = [StageWrapper(stage) for stage in self.config.stage_sequence]
#     for wrapped_stage in wrapped_stages:
#       data_in = wrapped_stage.process(data_in)

#     return data_in

#   def impute_stage_io(self, input_path: DataPath, output_path: DataPath) -> None:

#     current_input = input_path
#     for stage in self.config.stage_sequence:
#       current_output = output_path / stage.get_stage_name()
#       stage.config.input_path = current_input
#       stage.config.output_path = current_output
#       current_input = current_output


#   def transform_batch(self, input_path: DataPath, output_path: DataPath) -> None:

#     """
#     transformes a batch of files and persist soutput

#     """
    
#     self.impute_stage_io(self, input_path, output_path)

#     for stage in self.config.stage_sequence:
#       loggger.info(f"running stage: {stage.get_stage_name()}")
#       stage.run_from_config()

#   def fit(self, data: t.Any) -> t.Any:
#     """
#     Fits fittable stages in the pipeline

#     """
#     data_in = (input_data)
#     wrapped_stages = [StageWrapper(stage) for stage in self.config.stage_sequence]
#     for wrapped_stage in wrapped_stages:
#       data_in = wrapped_stage.fit_and_process(data_in)

#     return data_in

#   def save(self, output_path: DataPath) -> None:
#     """
#     Persists configuration yaml

#     output_path: path to destination folder

#     """
#     logger.info(f"{self.__class__.__name__} configuration persisted at {output_path}")
#     (output_path/ "pipeline").write_bytes(bytes(self.config.as_yaml().encode("utf-8")))


#   @classmethod
#   def load(cls, file: DataPath) -> MLPipeline:
#     logger.info(f"{self.__class__.__name__} configuration loaded from {output_path}")
#     return cls(cls.get_config_parser().from_file(file))

#   @classmethod
#   def get_config_parser(cls) -> ConfigHandler:
#     """
#     Returns an instance of the stage's configuration parser class

#     """
#     return ConfigHandler()
    

