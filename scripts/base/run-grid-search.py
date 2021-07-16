# Runs a grid search on configuration parameters for a Scoring stage object. This include any of the usual ML model hyperparameters

import os
import sys
import argparse
import logging
import itertools
import collections
import typing as t
import json
from pathlib import Path

from fontai.config.core import BasePipelineTransformConfig
from fontai.runners.base import FittableTransform
from fontai.runners.stages import Preprocessing, Ingestion, Scoring

import mlflow
import numpy as np
import strictyaml as yml

logger = logging.getLogger(__name__)

Path("logs").mkdir(parents=True, exist_ok=True)

def flatten_dict(
  d: t.Dict, 
  parent_key: str = '', 
  sep: str = '.'):
    """Flatten nested dicts. This function is useful when logging a potentially nested hyperparameter dict to MLflow, which doesn't support nested parameter logging.
    
    Args:
        d (t.Dict): dictionary to flatten
        parent_key (str, optional): Preffix to keys in flattened dict
        sep (str, optional): Separator between flattened keys
    
    Returns:
        t.Dict: flattened dict
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def build_parameter_grid(json_file: str) -> t.Dict:
  """Builds a parameter grid from a JSON file's path pointing to a grid definition
  
  Args:
      json_file (str): File path
  
  Returns:
      t.List: List of dictionaries defining a grid point
  """
  with open(json_file,"r") as f:
    json_grid = json.loads(f.read())
  #
  key_value_pairs = [(key, json_grid[key]) for key in json_grid]
  values = [value for key, value in key_value_pairs]
  keys = [key for key, value in key_value_pairs]
  #
  grid = list(itertools.product(*values))
  #
  # add keys to grid points
  dict_grid = [{key: val for key,val in zip(keys,point)} for point in grid]
  return dict_grid

def point_to_config(
  base_config: yml.YAML, 
  validator: yml.Map,
  point: t.Dict) -> BasePipelineTransformConfig:
  """Produce a valid configuration YAML from a point in the hyperparameter search grid 
  
  Args:
      base_config (BasePipelineTransformConfig): Base training configurations
      validator (yml.Map): schema validator for resulting grid point configuration
      point (t.List): Hyperparameter point as a dictionary where keys are paths to parameters in the YAML file using dot notation, and values are specific parameter values to try, e.g. {'training.optimizer.kwargs.learning_rate': 0.001}
  
  Yields:
      t.Generator[BasePipelineTransformConfig, None, None]: Training configuration
  """
  config_data = base_config.copy().data
  #
  for param in point:
    current = config_data
    value = point[param]
    path = param.split(".")
    path_depth = len(path)
    #
    for path_index in range(path_depth-1):
      path_edge = path[path_index]
      try:
        edge = int(path_edge)
      except ValueError as e:
        edge = path_edge
      current = current[edge]
    #
    # print(f"current: {current}, key: {path[-1]}, value: {value}")
    current[path[-1]] = value
  return yml.as_document(config_data, validator)


def run(args):
  
  parser = argparse.ArgumentParser(description = "Performs a grid search on a subset of parameters in the YAML configuration file for an instance of a scoring stage. All parameters in the YAML file are searchable and the grid is specified in a JSON file with a specific syntax, see below for details.")
  parser.add_argument(
      '--base-config',
      dest='base_config_file',
      required = True,      
      help=
      """
      path to YAML file that defines training execution, including model architecture.
      """)
  parser.add_argument(
      '--grid-file',
      dest='grid_file',
      required = True,      
      help=
      """
      Path to JSON file with grid specification. Any parameter in the base config file can be grid-searched by using dot notation in the JSON grid file, e.g. {'training.optimizer.kwargs.learning_rate': [0.001, 0.0001]} would define a grid on the learning rate; list entries in the base configuration can be accessed using integers as strings, e.g. {model.encoder.kwargs.layers.1.kwargs.units: [1024, 1280]}. The final grid is the Cartesian product of every entry in the JSON file.
      """)
  parser.add_argument(
      '--experiment-name',
      dest='experiment_name',
      required = True,      
      help=
      """
      Name to group runs in MLFlow tracking.
      """)

  args, _ = parser.parse_known_args(args)

  logging.basicConfig(filename=Path("logs") / f"grid-search-{args.experiment_name}.log", level=logging.DEBUG, filemode = "w")

  print(f"Redirecting logs to logs/grid-search-{args.experiment_name}.log")

  config_parser = Scoring.get_config_parser()
  schema_validator = config_parser.get_config_schema()
  base_config = config_parser.from_file(args.base_config_file).yaml

  grid = build_parameter_grid(args.grid_file)

  mlflow.set_experiment(args.experiment_name)

  for run_id, point in enumerate(grid):
    run_config_yaml = point_to_config(base_config, schema_validator, point)
    with mlflow.start_run(run_id = f"{args.experiment_name}-{run_id}") as run:
      mlflow.log_params(flatten_dict(point))
      Scoring.fit_from_config_file(config_parser.instantiate_config(run_config_yaml))
    #print(config.as_yaml())

  
if __name__ == "__main__":

  run(sys.argv)
