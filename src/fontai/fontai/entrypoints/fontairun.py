#!python
"""This script runs a single MLOps pipeline stage according to provided parameters. Use --help for more details.
"""
import sys
import os
import argparse
import logging
import datetime
import traceback
from pathlib import Path

logger = logging.getLogger("fontai")

from fontai.runners.base import FittableTransform
import fontai.runners.stages as stages

from tensorflow.config import get_visible_devices
from tensorflow.python.framework.errors_impl import UnknownError as unknown_tf_error

Path("logs").mkdir(parents=True, exist_ok=True)

class RunNotAllowedError(Exception):
  pass

class StageRunner(object):

  """Parses CLI arguments and executed an ML lifecycle stage.
  
  Attributes:
      parser (argparse.ArgumentParser): Argument parser
      schema_docstring (str): A compilation of docstrings from available runner classes and their parsers. It is meant to be displayed in the command line using the `--help` parameter.
      stage_runners (dict): Maps strings to runner clases for the `--stage` argument
  """
  stage_runners = {getattr(stages, name).get_stage_name() : getattr(stages, name) for name in stages.__all__}

  
  schema_docstring = "\n\n\n". join([f"{key}: \n\t{val.get_config_parser().get_config_schema.__doc__}" for key, val in stage_runners.items()])
  
  parser = argparse.ArgumentParser(
    description = "Runs a single ML processing stage with its execution specified by a YAML configuration file. See below for details of configuration schema.",
    formatter_class = argparse.RawTextHelpFormatter)
  parser.add_argument(
      '--stage',
      dest='stage',
      required = True,      
      help=
      f"""
      One of: {", ".join(list(stage_runners.keys()))}
      """)
  parser.add_argument(
      '--config-file',
      dest='config_file',
      required = True,      
      help=
      f"""path to YAML file that defines the execution. Configuration schemas are as follows:
\n
{schema_docstring}
      """)
  parser.add_argument(
      '--fit',
      dest='fit',
      action = 'store_true',      
      help=
      """
      If present (and if --stage is 'scoring'), fits the provided model, otherwise scores input data and saves the scored examples to the output_path parameter in the config file.
      """)
  parser.add_argument(
      '--load-model',
      dest='load_model',
      action = 'store_true',      
      help=
      """
      If set, the model is loaded from model_path instead of building it from scratch. This is useful for retraining an existing model.
      """)
  parser.add_argument(
      '--run-name',
      dest='run_name',
      default = None,      
      help=
      """
      Name of previously logged MLFlow run that is to be continued.
      """)
  parser.add_argument(
      '--logging-level',
      dest='logging_level',
      default = "INFO",      
      help=
      """
      Logging level. Defaults to 'INFO'
      """)

  @classmethod
  def run(cls, args=None):
    """Run specified ML processing operation with the given configuration.
    
    Args:
        args (t.List[str]): CLI arguments
    
    Raises:
        TypeError: If --fit is provided when stage is not fittable.
    """
    if args is None:
      args = sys.argv
      
    args, _ = cls.parser.parse_known_args(args)


    # set up logging
    logging_level = getattr(logging, args.logging_level)

    if os.environ.get("CONTAINER_ENV", "false") == "true":
      # if executing in a container, log to stdout
      logging.basicConfig(level=logging_level)
      logger.info(f"sys.argv input: {sys.argv}")
      logger.info(f"Visible CUDA devices: {get_visible_devices()}")

    if os.environ.get("CONTAINER_ENV", "false") != "true":
      logfile = f"{args.stage}-{datetime.datetime.now()}.log"
      logging.basicConfig(filename=Path("logs") / logfile, level=logging_level, filemode = "w")
      print(f"Redirecting logs to logs/{logfile}")

    # sanitize parameters
    try:
      stage_class = cls.stage_runners[args.stage]
    except KeyError as e:
      raise ValueError(f"stage must be one of {', '.join(list(cls.stage_runners.keys()))}")

    if args.fit and not issubclass(stage_class, FittableTransform):
      raise RunNotAllowedError(f"stage {args.stage} is not fittable.")

    if args.load_model and args.stage != "scoring":
      raise RunNotAllowedError("--load-model is only supported for input scoring or model fitting.")

    if args.stage == "scoring" and not args.fit and not args.load_model:
      raise RunNotAllowedError("Scoring input data can only be done with an existing (i.e. saved) model. Use the --load-model flag to load one from model_path.")


    # run
    if args.fit:
      try:
        stage_class.fit_from_config_file(args.config_file, run_id=args.run_name, load_from_model_path=args.load_model)
      except unknown_tf_error as e:
        raise Exception(f"{traceback.format_exc()}.\n\n *** If you get a cuDNN error, try doing: export TF_FORCE_GPU_ALLOW_GROWTH=true *** \n\n")
    else:
      stage_class.run_from_config_file(args.config_file, load_from_model_path=args.load_model)
  