# example run: python scripts/run_single_stage.py --config-file config/parameters/local-preprocessing.yaml --stage ingestion
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true' #this is needed to run models on GPU

import sys
import argparse
import logging
from pathlib import Path

from fontai.pipeline.base import FittableTransform
from fontai.pipeline.stages import LabeledExampleExtractor, FontIngestion, Predictor

logger = logging.getLogger(__name__)

Path("logs").mkdir(parents=True, exist_ok=True)

def process_files(args):
  stage_classes = {
    "ingestion": FontIngestion,
    "preprocessing": LabeledExampleExtractor,
    "prediction": Predictor
  }
  
  parser = argparse.ArgumentParser(description = "Runs single ML processing stage specified by a YAML file.")
  parser.add_argument(
      '--config-file',
      dest='config_file',
      required = True,      
      help=
      """
      path to YAML file that defines the execution.
      """)
  parser.add_argument(
      '--stage',
      dest='stage',
      required = True,      
      help=
      """
      One of 'ingestion', 'preprocessing' or 'prediction'
      """)
  parser.add_argument(
      '--fit',
      dest='fit',
      action = 'store_true',      
      help=
      """
      If true, fits stage.
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

  stage_class.fit_from_config_file(args.config_file) if args.fit else stage_class.run_from_config_file(args.config_file)
  
if __name__ == "__main__":

  process_files(sys.argv)
