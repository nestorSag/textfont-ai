from abc import ABC
import logging
import typing as t
import pickle

from fontai.core.base import BaseConfigHandler, MLPipelineStage, FittableStage
from fontai.config.pipeline import Config, ConfigHandler
from fontai.core.io import DataPath

logger = logging.getLogger(__name__)

class MLPipeline(object):
  """
    Wrapper class for a sequence of ML pipeline stages; can both fit stages that are fittable and process both batch and in-memory data

    config: Configuration object that defines the stages' execution

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
