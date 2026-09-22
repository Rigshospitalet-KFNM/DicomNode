# Python standard library
from argparse import _SubParsersAction, Namespace, ArgumentParser
import inspect
import importlib
import importlib.util
import os
import sys
from pathlib import Path
from textwrap import dedent
from typing import Type

# Third party Packages

# Dicomnode packages
from dicomnode.config import DicomnodeConfig, config_from_raw
from dicomnode.lib.io import File
from dicomnode.server.nodes import AbstractPipeline

HELP_MESSAGE = dedent("""
This programs takes a python file with a single AbstractPipeline implementation.
Then this program will open that pipeline. You can also pass a config file that
takes precedence over the configuration of the pipeline.
""")

def load_config_from_python_file(path):
  config_module = load_module(path)

  if not hasattr(config_module, 'CONFIG'):
    raise Exception(f"python module at {path} doesn't have a CONFIG attribute")

  if not isinstance(config_module.CONFIG, DicomnodeConfig):
    raise Exception(f"CONFIG in {path} is not DicomnodeConfig type object")

  return config_module.CONFIG


def load_config_from_file(path: Path, PipeLine: Type[AbstractPipeline]) -> DicomnodeConfig:

  return config_from_raw()

def load_module(pipeline_file_str):
  pipeline_path = Path(pipeline_file_str).absolute()
  pipeline_module = pipeline_path.name.split(".")[0]

  # There's some consideration, that we should use the name and
  spec = importlib.util.spec_from_file_location(pipeline_module, pipeline_path)
  if spec is None:
    raise FileNotFoundError(f"Unable to find python module at {pipeline_path}")

  module = importlib.util.module_from_spec(spec)
  sys.modules[pipeline_module] = module
  if spec.loader is None:
    raise Exception("The loader is None??")

  # This ensures that imports from the module works
  sys.path.insert(0, os.getcwd())
  sys.path.insert(0, str(pipeline_path.parent))

  spec.loader.exec_module(module)
  return module

def get_pipeline_implementations(module):
  return [
    obj for _name, obj in inspect.getmembers(module, inspect.isclass)
      if issubclass(obj, AbstractPipeline) and obj is not AbstractPipeline and
        obj.__module__ == module.__name__
  ]

def get_parser(subparser: _SubParsersAction): #pragma: no cover
  _, _, tool_name = __name__.split(".")

  module_parser: ArgumentParser = subparser.add_parser(tool_name, help=HELP_MESSAGE)

  module_parser.add_argument("pipeline_file", type=str, help="Path to python file that contains your AbstractPipeline")
  module_parser.add_argument("config_file", type=str, default="", help="Path to a config file, that will pass into pipeline. Can be of .json, .py (define CONFIG) member")
  module_parser.add_argument("-v", "--verbose", action='store_true')

def entry_func(args: Namespace): #pragma: no cover
  pipeline_path: str = args.pipeline_file

  module = load_module(pipeline_path)

  pipelines = get_pipeline_implementations(module)

  if len(pipelines) == 0:
    raise Exception(f"Unable to find an AbstractPipeline implementation in {pipeline_path}")
  elif len(pipelines) > 1:
    raise Exception(f"Found multiple pipeline implementations in {pipeline_path}. To run it - make sure you only have 1")

  config_path = Path(args.config_file)

  if config_path.exists():
    pass
  elif args.verbose:
    print("Config file not found")

  pipeline = pipelines[0]()

  pipeline.open()