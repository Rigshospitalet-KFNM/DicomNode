"""Script part of the omnitool used to send a C-store"""

__author__ = "Christoffer Vilstrup Jensen"

# Python standard library
from argparse import _SubParsersAction, Namespace
from pathlib import Path

# Third party packages

# Dicomnode packages
from dicomnode.dicom.dimse import send_images, Address
from dicomnode.lib.exceptions import CouldNotCompleteDIMSEMessage
from dicomnode.data_structures.image_tree import StudyTree
#from dicomnode.lib.io import load_private_tags_from_args
from dicomnode.lib.utils import str2bool


def get_parser(subparser : _SubParsersAction):
  """Function to generate the arguments to the tool

  Args:
    subparser (_SubParsersAction): This is the parser that arguments should
                                   be added to.
  """
  _, _, tool_name = __name__.split(".")
  module_parser = subparser.add_parser(tool_name,
                                       help="Sends a C_STORE DIMSE Message with <dicom file>")
  module_parser.add_argument('ip', type=str, help="IP of SCP")
  module_parser.add_argument('port', type=int, help="Port of SCP")
  module_parser.add_argument('SCP_AE', type=str, help="The AE title of the SCP")
  module_parser.add_argument('SCU_AE', type=str, help="The AE title of the SCU")
  module_parser.add_argument('dicomfile', type=Path,
                             help="Path to dicom file to be save")
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
  #private_tags = load_private_tags_from_args(args) # note imlimented
  study_tree = StudyTree()
  study_tree.discover(args.dicomfile)
  address = Address(args.ip, args.port, args.SCU_AE)

  try:
    send_images(args.SCP_AE, address, study_tree)
  except CouldNotCompleteDIMSEMessage:
    print(f"Could not connect to the SCP with the following inputs:\nIP: \
          {args.ip}\nPort: {args.port}\nSCP AE: {args.SCP_AE}\n SCU AE:{args.SCU_AE}")
