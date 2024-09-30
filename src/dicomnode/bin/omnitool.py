""" This is the main entry point for the omnitool, which is a collections of scripts involving dicom images


"""
__author__ = "Christoffer Vilsturp Jensen"

import argparse
import importlib
from dicomnode import tools

def entry_func():
  parser = argparse.ArgumentParser(description="This is an omnitool for dicom communication")
  subparsers = parser.add_subparsers(help="help", dest="command")
  modules = {}

  for tool in filter(lambda module: module[0] != '_', dir(tools)):
    module = importlib.import_module(f"dicomnode.tools.{tool}")
    modules[tool] = module
    try:
      module_parser = module.get_parser(subparsers)

    except AttributeError as E:
      ErrorMessage = f"The module {tool} has no method \'get_parser\'"
      raise NotImplementedError(ErrorMessage)

  args = parser.parse_args()

  modules[args.command].entry_func(args)
