"""This is a script part of the omnitool for displaying a dicomfile"""

__author__ = "Chirstoffer Vilstrup Jensen"

from argparse import _SubParsersAction, Namespace

from pprint import pprint
from pathlib import Path
from dicomnode.lib.utils import str2bool
from dicomnode.lib.io import load_dicom, load_private_tags_from_args


def get_parser(subparser : _SubParsersAction):
  """Function to generate the arguments to the tool

  Args:
    subparser (_SubParsersAction): This is the parser that arguments should
                                   be added to.
  """
  _, _, tool_name = __name__.split(".")
  module_parser = subparser.add_parser(tool_name,
                                       help="Displays a dicom file")
  module_parser.add_argument('dicomfile',
                             type=Path,
                             nargs='*',
                             help="Path to dicom file to be shown")
  module_parser.add_argument('--privatetags',
                             type=Path,
                             help="Path to .dlc file with private tags")
  module_parser.add_argument('--strictParsing',
                             type=str2bool,
                             nargs='?',
                             const=False,
                             default=False,
                             help="Stop if a private tag is not parsed correctly")

def entry_func(args : Namespace):
  """This is the function that should does the work of the tool.

  Args:
      args (Namespace): The user arguments

  """
  private_tags = load_private_tags_from_args(args)
  for dicomfile in args.dicomfile:
    pprint(load_dicom(dicomfile, private_tags=private_tags))
