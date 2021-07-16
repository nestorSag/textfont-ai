# example run: python scripts/run_single_stage.py --config-file config/parameters/local-preprocessing.yaml --stage ingestion
import sys
import argparse
import logging
from pathlib import Path

from fontai.runners.pipeline import Pipeline

logger = logging.getLogger(__name__)

Path("logs").mkdir(parents=True, exist_ok=True)

schema_docstring = f"Pipeline: \n\t{Pipeline.get_config_parser().get_config_schema.__doc__}"

def run(args):
  
  parser = argparse.ArgumentParser(
    description = "Runs a sequence of ML processing stages defined by the pipeline's YAML file",
    formatter_class = argparse.RawTextHelpFormatter)

  parser.add_argument(
      '--config-file',
      dest='config_file',
      required = True,      
      help=
      f"""
path to YAML file that defines pipeline execution. Configuration schema is as follows:
{schema_docstring}
      """)

  args, _ = parser.parse_known_args(args)

  logging.basicConfig(filename=Path("logs") / "pipeline.log", level=logging.INFO, filemode = "w")

  print(f"Redirecting logs to logs/pipeline.log")

  Pipeline.run_from_config_file(args.config_file)
  
if __name__ == "__main__":

  run(sys.argv)
