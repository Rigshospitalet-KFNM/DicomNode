"""This "module" is the tool kit for anonymizing a set of dicom studies

`python3 omnitool --help`


"""

# Python standard library
from argparse import _SubParsersAction, Namespace
from pathlib import Path
from shutil import rmtree

# Third party packages

# Dicomnode packages
from dicomnode.dicom.anonymization import anonymize_dicom_tree, BASE_ANONYMIZED_PATIENT_NAME
from dicomnode.data_structures.image_tree import DicomTree, IdentityMapping, _PPrefix
from dicomnode.lib.utils import str2bool

def get_parser(subparser : _SubParsersAction):
  """Function to generate the arguments to the tool

  Args:
    subparser (_SubParsersAction): This is the parser that arguments should
                                   be added to.
  """
  _, _, tool_name = __name__.split(".")
  module_parser = subparser.add_parser(tool_name, help="Anonymizes a file or Directory")
  module_parser.add_argument("dicom_path", type=Path, help="Path to directory or dicomFile")
  module_parser.add_argument(
    '--keepuids', type=str2bool, nargs='?', const=False, default=False,
      help="toggle to retain SOPInstanceID StudyUID and SeriesUID")
  module_parser.add_argument('--key', type=Path, help="Path to key file\
                              for anonymization,this files will contain\
                              Personal information!")
  module_parser.add_argument('--pidpf', type=str, default=_PPrefix, help="\
                             Prefix for PatientID, so anonymized PatientID\
                              will be <pidpf>XXXX where X is the patient number")
  module_parser.add_argument('--pnpf', type=str, default=BASE_ANONYMIZED_PATIENT_NAME,
                             help="Prefix for PatientName, so anonymized PatientName\
                                will be <pnpf>XXXX where X is the patient number")
  module_parser.add_argument('--sid', type=str, default="", help="Overwrites\
                              the StudyID with <sid>XXXX where X is the patient number")
  module_parser.add_argument('--overwrite', type=str2bool, nargs='?', const=False, default=False,
      help="Delete the directory / file at the destination")

def entry_func(args : Namespace):
  """This is the function that should does the work of the tool.

  Args:
      args (Namespace): The user arguments

  Raises:
      FileNotFoundError: when args.DicomPath is not a file or Directory
      FileExistsError: _description_
      FileExistsError: _description_
  """
  # This is first to find the DicomPath to fail fast.
  if args.dicom_path.is_file():
    target = args.dicom_path.parent / ("anon_" + args.dicom_path.name)
  elif args.DicomPath.is_dir():
    target = args.dicom_path.parent / ("anon_" + args.dicom_path.name)
  else:
    raise FileNotFoundError("Dicom path is not a file or Directory")

  if args.key and args.key.exists():
    key_removal_command = f"rm {str(args.key)}"
    raise FileExistsError(f"Key file already exists, run: {key_removal_command}")

  if target.exists():
    if args.overwrite:
      if target.is_file():
        target.unlink()
      elif target.is_dir():
        rmtree(target)
    else:
      error_message = f"The Target file {str(target)} Exists!"
      raise FileExistsError(error_message)

  tree = DicomTree()
  tree.discover(args.DicomPath)


  identity_mapping = IdentityMapping()
  identity_mapping.fill_from_DicomTree(
    tree,
    patient_prefix=args.pidpf,
    change_UIDs=not args.keepuids
  )

  tree.map(anonymize_dicom_tree(
    identity_mapping,
    PatientName=args.pnpf,
    StudyID=args.sid
  ), identity_mapping)

  tree.save_tree(target)
