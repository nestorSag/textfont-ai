#!python
"""This script runs a single MLOps pipeline stage according to provided parameters. Use --help for more details.
"""
import sys
import os
import argparse
import logging
import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

from fontai.runners.base import FittableTransform
from fontai.runners.stages import Preprocessing, Ingestion, Scoring, Deployment


Path("logs").mkdir(parents=True, exist_ok=True)

class StageRunner(object):

  """Parses CLI arguments and executed an ML lifecycle stage.
  
  Attributes:
      parser (argparse.ArgumentParser): Argument parser
      schema_docstring (str): A compilation of docstrings from available runner classes and their parsers. It is meant to be displayed in the command line using the `--help` parameter.
      stage_classes (dict): Maps strings to runner clases for the `--stage` argument
  """
  
  stage_classes = {
    "ingestion": Ingestion,
    "preprocessing": Preprocessing,
    "scoring": Scoring,
    "deployment": Deployment
  }
  
  schema_docstring = "\n\n\n". join([f"{key}: \n\t{val.get_config_parser().get_config_schema.__doc__}" for key, val in stage_classes.items()])
  
  parser = argparse.ArgumentParser(
    description = "Runs a single ML processing stage with its execution specified by a YAML configuration file. See below for details of configuration schema.",
    formatter_class = argparse.RawTextHelpFormatter)
  parser.add_argument(
      '--stage',
      dest='stage',
      required = True,      
      help=
      f"""
      One of: {", ".join(list(stage_classes.keys()))}
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
      '--run-name',
      dest='run_name',
      default = None,      
      help=
      """
      Name of previously logged MLFlow run that is to be continued.
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
    if os.environ.get("CONTAINER_ENV", "false") == "true":
      # if executing in a container, log to stdout
      handler = logging.StreamHandler(sys.stdout)
      handler.setLevel(logging.DEBUG)
      logger.addHandler(handler)
    else:
      logfile = f"{args.stage}-{datetime.datetime.now()}.log"
      logging.basicConfig(filename=Path("logs") / logfile, level=logging.DEBUG, filemode = "w")
      print(f"Redirecting logs to logs/{logfile}")

    # sanitize parameters
    try:
      stage_class = cls.stage_classes[args.stage]
    except KeyError as e:
      raise ValueError(f"stage must be one of {', '.join(list(cls.stage_classes.keys()))}")

    if args.fit and not issubclass(stage_class, FittableTransform):
      raise TypeError(f"stage {args.stage} is not fittable.")

    # run
    if args.fit:
      stage_class.fit_from_config_file(args.config_file, run_id=args.run_name)
    else:
      stage_class.run_from_config_file(args.config_file)
  
# if __name__ == "__main__":

#   StageRunner.run(sys.argv)
