# example run: python scripts/run_single_stage.py --config-file config/parameters/local-preprocessing.yaml --stage ingestion
import sys
import argparse
import logging
from pathlib import Path

from fontai.pipeline.pipeline import Pipeline

logger = logging.getLogger(__name__)

Path("logs").mkdir(parents=True, exist_ok=True)

def process_files(args):
  
  parser = argparse.ArgumentParser(description = "Runs a sequence of ML processing stages defined by YAML files")

  parser.add_argument(
      '--config-file',
      dest='config_file',
      required = True,      
      help=
      """
      path to YAML file that defines pipeline execution.
      """)
  args, _ = parser.parse_known_args(args)

  logging.basicConfig(filename=Path("logs") / "pipeline.log", level=logging.INFO, filemode = "w")

  print(f"Redirecting logs to logs/pipeline.log")

  Pipeline.run_from_config_file(args.config_file)
  
if __name__ == "__main__":

  process_files(sys.argv)
