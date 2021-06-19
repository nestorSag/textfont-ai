# example run: python scripts/preprocessing/run_file_processing.py --config-file config/parameters/local-preprocessing.yaml
import sys
import argparse
import logging
from pathlib import Path

from fontai.pipeline.stages import LabeledExampleExtractor

logger = logging.getLogger(__name__)

Path("logs").mkdir(parents=True, exist_ok=True)

logging.basicConfig(filename=Path("logs") / "preprocessing.log", level=logging.INFO, filemode = "w")

def process_files(args):
  # Run file processing Beam pipeline as defined by the passed configuration YAML file
  
  parser = argparse.ArgumentParser(description = "Maps output from ingestion stage to tensorflow Record files for model consumption.")
  parser.add_argument(
      '--config-file',
      dest='config_file',
      required = True,      
      help=
      """
      path to YAML file that defines the execution of the file processing pipeline.
      """)

  print(f"Redirecting logs to logs/preprocessing.log")

  args, _ = parser.parse_known_args(args)
  
  LabeledExampleExtractor.run_from_config_file(args.config_file)

if __name__ == "__main__":

  process_files(sys.argv)
