"""
  This module contains a basic orchestrator for the execution of sequential data transformation stages.
"""
from __future__ import annotations
import typing as t
import types
from fontai.config.pipeline import Config as PipelineConfig, ConfigHandler as PipelineConfigHandler
from fontai.pipeline.base import ConfigurableTransform, FittableTransform
from fontai.config.core import BasePipelineTransformConfig


class ManyToManyTransform(object):

  """Helper class to execute one-to-many many-to-many transformations in the pipeline
  
  Attributes:
      core_transform (ConfigurableTransform): Core transformer class
  """

  def __init__(self, core_transform):
    self.core_transform = core_transform

  def transform(self, data: t.Any):
    """Outputs a generator of transformed elements
    
    Args:
        data (t.Any): Input data
    
    Yields:
        t.Any: individual outputs
    """
    for elem in self.to_generator(data):
      for out in self.to_generator(self.core_transform.transform(elem)):
        yield out

  def to_generator(self, data):
    if not isinstance(data, types.GeneratorType):
      return iter((data,))
    else:
      return data

class Pipeline(ConfigurableTransform):

  """Pipeline class to execute a sequence of ConfigurableTransforms; this allows to perform the whole set of transformations from raw data to trained model
  
  Attributes:
      configs (t.List[BasePipelineTransformConfig]): Sequence of configuration files, one per tranformation in the pipeline
      streaming_pipeline (t.List[ConfigurableTransform]): List of instantiated transforms
      transforms (type): types of transforms in the pipeline, in order
  """

  def __init__(self, transforms: t.List[type], configs: t.List[BasePipelineTransformConfig]):
    """Summary
    
    Args:
        transforms (t.List[type]): List of transformations in the pipeline
        configs (t.List[BasePipelineTransformConfig]): List of parsed configurations, one per stage in the pipeline
    """
    self.transforms = transforms
    self.configs = configs

    self.streaming_pipeline = [
      ManyToManyTransform(core_transform = transform.from_config_object(config)) for transform, config in zip(self.transforms, self.configs)]

  def transform(self, data: t.Any) -> t.Any:
    
    out = data
    for streaming_transform in self.streaming_pipeline:
      out = streaming_transform.transform(out)

    return out

  @classmethod
  def from_config_object(cls, config: PipelineConfig) -> Pipeline:
    return cls(config.stages, config.configs)

  @classmethod
  def run_from_config_object(cls, config: PipelineConfig) -> None:
    pipeline = cls.from_config_object(config)
    for t, config in zip(pipeline.transforms, pipeline.configs):
      t.run_from_config_object(config)

  @classmethod
  def get_config_parser(cls) -> PipelineConfigHandler:
    return PipelineConfigHandler()

  def fit(self, data: t.Any) -> FittableTransform:
    raise NotImplementedError("This class does not implement a fit() method")

  @classmethod
  def fit_from_config_object(cls, config: PipelineConfig) -> FittableTransform:
    pipeline = cls.from_config_object(config)
    for t, config in zip(pipeline.transforms, pipeline.configs):
      if issubclass(t, FittableTransform):
        t.fit_from_config_object(config)
      else:
        t.run_from_config_object(config)