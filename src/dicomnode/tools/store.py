"""Script part of the omnitool used to send a C-store"""

__author__ = "Christoffer Vilstrup Jensen"

from argparse import _SubParsersAction, Namespace
from pathlib import Path
from dicomnode.lib.dimse import send_images
from dicomnode.lib.exceptions import CouldNotCompleteDIMSEMessage

from dicomnode.lib.utils import str2bool
from dicomnode.lib.io import load_dicom, load_private_tags_from_args



def get_parser(subparser : _SubParsersAction):
  _, _, tool_name = __name__.split(".")
  module_parser = subparser.add_parser(tool_name, help="Sends a C_STORE DIMSE Message with <dicom file>")
  module_parser.add_argument('ip', type=str, help="IP of SCP")
  module_parser.add_argument('port', type=int, help="Port of SCP")
  module_parser.add_argument('SCP_AE', type=str, help="The AE title of the SCP")
  module_parser.add_argument('SCU_AE', type=str, help="The AE title of the SCU")
  module_parser.add_argument('dicomfile', type=Path, help="Path to dicom file to be save")
  module_parser.add_argument('--privatetags', type=Path, help="Path to .dlc file with private tags")
  module_parser.add_argument('--strictParsing', type=str2bool, nargs='?', const=False, default=False, help="Stop if a private tag is not parsed correctly")

def entry_func(args : Namespace):
  private_tags = load_private_tags_from_args(args)
  dicom_object = load_dicom(args.dicomfile, private_tags)
  try:
    c_store_resp = send_images(args.ip, args.port, args.SCP_AE, args.SCU_AE, dicom_object)
  except CouldNotCompleteDIMSEMessage:
    print(f"Could not connect to the SCP with the following inputs:\nIP: {args.ip}\nPort: {args.port}\nSCP AE: {args.SCP_AE}\n SCU AE:{args.SCU_AE}")
