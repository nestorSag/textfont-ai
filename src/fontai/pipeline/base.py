"""This module contains the base interfaces implemented by ML pipeline stages and pipeline objects.

"""

import logging
import typing
import sys
import typing as t


from fontai.config.base import BaseConfig

logger = logging.Logger(__name__)
  

class ConfigurableTransform(ABC):

  """Interface for configurable tranformations; they can be instantiated and run from YAML configuration files.
  """

  @classmethod
  @abstractmethod
  def from_config(cls, config: BaseConfig):
    """Instantiate class from a configuration object
    
    Args:
        config (BaseConfig): Config object parsed from a YAML file
    """
    pass

  @classmethod
  def parse_config(cls, path: str) -> BaseConfig:
    """Parse a YAML configuration file and create an instance inheriting from BaseConfig
    
    Args:
        path (str): Path to the YAML configuration file
    
    Returns:
        BaseConfig: Instantiated Config instance
    """
    return self.get_config_parser().from_file(BytestreamPath(path))

  @classmethod
  def from_config_file(cls, path: str):
    """Create a ConfigurableTransform instance from a YAML configuration file
    
    Args:
        path (str): Path to the YAML configuration file
    
    Returns:
        ConfigurableTransform: Instance created from configuration file.
    """
    return cls.from_config(cls.parse_config(path))

  @classmethod
  @abstractmethod
  def run_from_config_file(cls, path: str) -> None:
    """Instantiate a ConfigurableTransform instance and process a file batch from storage, saving outputs to storage, using storage locations specified in the YAML configuration file.
    
    Args:
        path (str): Path to the YAML configuration file
    """
    pass

  @classmethod
  @abstractmethod
  def get_config_parser(cls) -> BaseConfigHandler:

    """
    Returns an instance of the transform's configuration parser class
    
    """
    pass






class MLPipelineTransform(ConfigurableTransform, ABC):
  """
    Interface implemented by runnable ML stage objects. They are initialised by passing an execution configuration object and can perform batch transforming according to it; they can also do stream transforming using the `transform` method.

    config: Execution configuration object
  """

  @classmethod
  def run_from_config_file(cls, path) -> None:

    config = cls.parse_config(path)
    if config.reader is None or config.writer is None:
      raise TypeError("Configuration object does not specify reader and writer instances.")
    cls.from_config(config).transform_batch(config.reader,config.writer)

  @abstractmethod
  def transform(self, data: t.Any) -> t.Generator[t.Any]:
    """
    transforme a single input instance
    
    Args:
        data (t.Any): Input data
    
    """
    pass


  @abstractmethod
  def transform_batch(self, reader: BatchReader, writer: BatchWriter) -> None:
    """
    transformes a batch of files and persist output
    
    Args:
        reader (BatchReader): Object inheriting from BatchReader
        writer (BatchWriter): Object inheriting from BatchWriter
    
    """
    pass


  @classmethod
  @abstractmethod
  def get_stage_name(cls) -> str:
    """
    Returns a string with the stage name
    
    """
    pass

  def save(self, output_folder: str) -> None:
    """
    Persists necessary data from which the instance can be loaded again.
    
    output_folder: BytestreamPath instance pointing to output folder.
    
    Args:
        output_folder (str): path to output folder in storage media
    
    """
    output_folder = BytestreamPath(output_folder)

    logger.info(f"{self.__class__.__name__} configuration persisted at {output_folder}")
    (output_folder / "config.yaml").write_bytes(bytes(self.config.yaml.as_yaml().encode("utf-8")))

  @classmethod
  def load(cls, input_folder: str) -> MLPipelineTransform:
    """
    Load an instance of this class
        
    Args:
        input_folder (str): Path to input folder
    
    Returns:
        MLPipelineTransform: Loaded instance
    
    """
    logger.info(f"{self.__class__.__name__} loaded from {input_folder}")
    return cls.get_config_parser().from_file(input_folder / "config.yaml")    







class FittableMLPipelineTransform(MLPipelineTransform,ABC):

  """Interface for pipeline transforms that can be fitted. Scoring is done using the 'transform' method.
  """

  def fit(self, data: t.Any) -> FittableMLPipelineTransform:
    """
    Fits the stage to the passed data

    """
    pass

  def save(self, output_folder: str) -> None:
    super().save(output_path)
    self.model.save(output_path)

  def load(self, input_folder: str) -> FittableMLPipelineTransform:
    loaded = super().load(input_folder)
    loaded.model = Model.load(input_folder)
    return loaded







class MLPipeline(object):
  """
  Wrapper class for a sequence of ML pipeline stages; can both fit stages that are fittable and process both batch and in-memory data
    
  Attributes:
      config (BaseConfig): Configuration object that defines the stages' execution
  
  """

  class StageWrapper(object):
    """
    Wrapper class for a single ML stage for one-to-many transformations.

    """

    def __init__(stage: MLPipelineStage):
      self.stage = stage

    def process(self, data: t.Generator[t.Any, None, None]) -> t.Generator[t.Any, None, None]:
      
      for elem_in in data:
        for elem_out in self.stage.process(elem_in):
          yield elem_out

    def fit(self, data: t.Generator[t.Any, None, None]) -> t.Generator[t.Any, None, None]:
      
      if isinstance(self.stage, FittableStage):
        self.stage.fit(data)

    def fit_and_process(self, data: t.Generator[t.Any, None, None]) -> t.Generator[t.Any, None, None]:
      
      self.fit(data)
      return self.process(data)


  def __init__(self, config: Config):
    self.config = config

  def run_from_config(self) -> None:

    self.transform_batch(self.config.input_path, self.config.output_path)

  def transform(self, input_data: t.Any) -> t.Any:
    """
    transformes a single data instance

    """

    data_in = (input_data)
    wrapped_stages = [StageWrapper(stage) for stage in self.config.stage_sequence]
    for wrapped_stage in wrapped_stages:
      data_in = wrapped_stage.process(data_in)

    return data_in

  def impute_stage_io(self, input_path: DataPath, output_path: DataPath) -> None:

    current_input = input_path
    for stage in self.config.stage_sequence:
      current_output = output_path / stage.get_stage_name()
      stage.config.input_path = current_input
      stage.config.output_path = current_output
      current_input = current_output


  def transform_batch(self, input_path: DataPath, output_path: DataPath) -> None:

    """
    transformes a batch of files and persist soutput

    """
    
    self.impute_stage_io(self, input_path, output_path)

    for stage in self.config.stage_sequence:
      loggger.info(f"running stage: {stage.get_stage_name()}")
      stage.run_from_config()

  def fit(self, data: t.Any) -> t.Any:
    """
    Fits fittable stages in the pipeline

    """
    data_in = (input_data)
    wrapped_stages = [StageWrapper(stage) for stage in self.config.stage_sequence]
    for wrapped_stage in wrapped_stages:
      data_in = wrapped_stage.fit_and_process(data_in)

    return data_in

  def save(self, output_path: DataPath) -> None:
    """
    Persists configuration yaml

    output_path: path to destination folder

    """
    logger.info(f"{self.__class__.__name__} configuration persisted at {output_path}")
    (output_path/ "pipeline").write_bytes(bytes(self.config.as_yaml().encode("utf-8")))


  @classmethod
  def load(cls, file: DataPath) -> MLPipeline:
    logger.info(f"{self.__class__.__name__} configuration loaded from {output_path}")
    return cls(cls.get_config_parser().from_file(file))

  @classmethod
  def get_config_parser(cls) -> ConfigHandler:
    """
    Returns an instance of the stage's configuration parser class

    """
    return ConfigHandler()
    

