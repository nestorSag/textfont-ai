# example run: python scripts/run_single_stage.py --config-file config/parameters/local-preprocessing.yaml --stage ingestion
import sys
import argparse
import logging
from pathlib import Path

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
      path to YAML file that defines the execution of the file processing pipeline.
      """)
  parser.add_argument(
      '--stage',
      dest='stage',
      required = True,      
      help=
      """
      One of 'ingestion', 'preprocessing' or 'prediction'
      """)
  args, _ = parser.parse_known_args(args)

  logging.basicConfig(filename=Path("logs") / f"{args.stage}.log", level=logging.INFO, filemode = "w")

  print(f"Redirecting logs to logs/{args.stage}.log")

  try:
    stage_classes[args.stage].run_from_config_file(args.config_file)
  except KeyError as e:
    print("stage must be one of 'ingestion', 'preprocessing' or 'prediction'")
  
if __name__ == "__main__":

  process_files(sys.argv)
