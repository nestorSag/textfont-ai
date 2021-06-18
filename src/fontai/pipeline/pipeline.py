"""
  This module contains a basic orchestrator for the execution of sequential data transformation stages.
"""

import typing as t
import types
from fontai.config.pipeline import Config as PipelineConfig, ConfigHandler as PipelineConfigHandler
from fontai.pipeline.base import ConfigurableTransform, FittableTransform


class Pipeline(ConfigurableTransform):

  """Pipeline class to execute a sequence of ConfigurableTransforms; this allows to perform the whole set of transformations from raw data to trained model
  
  Attributes:
      configs (t.List[BasePipelineTransformConfig]): Sequence of configuration files, one per tranformation in the pipeline
      streaming_pipeline (t.List[ConfigurableTransform]): List of instantiated transforms
      transforms (type): types of transforms in the pipeline, in order
  """
  
  class ManyToManyTransform(object):

    """Helper class to execute one-to-many many-to-many transformations in the pipeline
    
    Attributes:
        transform (ConfigurableTransform): Core transformer class
    """

    def __init__(self, transform):
      self.transform = transform

    def transform(self, data: t.Any):
      """Outputs a generator of transformed elements
      
      Args:
          data (t.Any): Input data
      
      Yields:
          t.Any: individual outputs
      """
      if not isinstance(data, types.GeneratorType):
        data = iter(data)
      for elem in data:
        for out in self.transform.transform(elem):
          yield out

  def __init__(self, transforms: t.List[type], configs: t.List[BasePipelineTransformConfig]):
    """Summary
    
    Args:
        transforms (t.List[type]): List of transformations in the pipeline
        configs (t.List[BasePipelineTransformConfig]): List of parsed configurations, one per stage in the pipeline
    """
    self.transforms = transforms
    self.configs = configs

    self.streaming_pipeline = [
      ManyToManyTransform(transform = transform.from_config_object(config)) for transform, config in zip(self.transforms, self.configs)]

  def transform(self, data: t.Any) -> t.Any:
    
    out = data
    for streaming_transform in self.streaming_pipeline:
      out = streaming_transform.transform(out)

    return out

  @classmethod
  def from_config_object(self, config: PipelineConfig) -> MultiTransform:
    return cls(config.transforms, config.configs)

  @classmethod
  def run_from_config_object(self, config: PipelineConfig) -> None:
    pipeline = cls.from_config_object(config)
    for t, config in zip(pipeline.transforms, pipeline.configs):
      t.run_from_config_object(config)

  @classmethod
  def get_config_parser(cls) -> BasePipelineTransformConfigHandler:
    return PipelineConfigHandler

  def fit(self, data: t.Any) -> FittableTransform:
    raise NotImplementedError("This class does not implement a fit() method")

  @classmethod
  def fit_from_config_object(self, config: PipelineConfig) -> FittableTransform:
    pipeline = cls.from_config_object(config)
    for t, config in zip(pipeline.transforms, pipeline.configs):
      if issubclass(t, FittableTransform):
        t.fit_from_config_object(config)
      else:
        t.run_from_config_object(config)