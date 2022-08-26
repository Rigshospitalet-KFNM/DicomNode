from argparse import _SubParsersAction, ArgumentParser, Namespace, ArgumentTypeError
import argparse

from pprint import pprint


from pathlib import Path
from dicomnode.lib.io import load_dicom, load_private_tags

def str2bool(v:str) -> bool:
  if isinstance(v, bool):
    return v
  if v.lower() in ['yes', 'true', 't', 'y', '1']:
    return True
  elif v.lower() in ['no', 'false', 'no', 'n','f', '0']:
    return False
  else:
    raise ArgumentTypeError("Boolean value expected")


def get_parser(subparser : _SubParsersAction):
  _, _, tool_name = __name__.split(".")
  module_parser = subparser.add_parser(tool_name, help="Displays a dicom file")
  module_parser.add_argument('dicomfile', type=Path, help="Path to dicom file to be shown")
  module_parser.add_argument('--privatetags', type=Path, help="Path to .dlc file with private tags")
  module_parser.add_argument('--strictParsing', type=str2bool, nargs='?', const=False, default=False, help="Stop if a private tag is not parsed correctly")

def entry_func(args : Namespace):
  privateTags = None
  if args.privatetags:
    load_private_tags(args.privatetags)

  #pprint(load_dicom(args.dicomfile, private_tags=privateTags))