# Python standard library
from argparse import _SubParsersAction, Namespace, ArgumentParser
from pathlib import Path
from textwrap import dedent
import importlib

# Third party Packages

# Dicomnode packages
from dicomnode.lib.io import FileType, verify_path

HELP_MESSAGE = dedent("""
This programs takes a python file and in that file a single AbstractPipeline
must be defined. Then this program will open that pipeline.
""")

def get_parser(subparser: _SubParsersAction): #pragma: no cover
  _, _, tool_name = __name__.split(".")

  module_parser: ArgumentParser = subparser.add_parser(tool_name, help=HELP_MESSAGE)

  module_parser.add_argument("pipeline_file", type=Path, help="Path to python file that contains your AbstractPipeline")

def entry_func(args: Namespace): #pragma: no cover
  pipeline_path: Path = args.pipeline_file

  main(pipeline_path)


def main(pipeline_path: Path):
  if not verify_path(pipeline_path, FileType.FILE):
    raise FileNotFoundError(f"{pipeline_path} is not a file!")
