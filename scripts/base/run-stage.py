# example run: python scripts/run_single_stage.py --config-file config/parameters/local-preprocessing.yaml --stage ingestion
import sys
import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

from fontai.runners.base import FittableTransform
from fontai.runners.stages import Preprocessing, Ingestion, Scoring


Path("logs").mkdir(parents=True, exist_ok=True)


def run(args):
  stage_classes = {
    "ingestion": Ingestion,
    "preprocessing": Preprocessing,
    "scoring": Scoring
  }

  schema_docstring = "\n\n\n". join([f"{key}: \n\t{stage_classes[key].get_config_parser().get_config_schema.__doc__}" for key in stage_classes])
  
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
      If present (and if stage is scoring), fits stage, otherwise perform scoring on input data.
      """)
  parser.add_argument(
      '--run-name',
      dest='run_name',
      default = None,      
      help=
      """
      Optional name for logged MLFlow run. Only used for model training.
      """)

  args, _ = parser.parse_known_args(args)

  logging.basicConfig(filename=Path("logs") / f"{args.stage}.log", level=logging.DEBUG, filemode = "w")

  print(f"Redirecting logs to logs/{args.stage}.log")

  try:
    stage_class = stage_classes[args.stage]
  except KeyError as e:
    print("stage must be one of 'ingestion', 'preprocessing' or 'prediction'")

  if args.fit and not issubclass(stage_class, FittableTransform):
    raise TypeError(f"stage {args.stage} is not fittable.")

  if args.fit:
    stage_class.fit_from_config_file(args.config_file, run_id=args.run_name)
  else:
    stage_class.run_from_config_file(args.config_file)
  
if __name__ == "__main__":

  run(sys.argv)
