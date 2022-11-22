
from email.policy import default
from pathlib import Path
from shutil import rmtree


from argparse import _SubParsersAction, Namespace
from dicomnode.lib.anonymization import anonymize_dicom_tree, BASE_ANONYMIZED_PATIENT_NAME
from dicomnode.lib.imageTree import DicomTree, IdentityMapping, _PPrefix


from dicomnode.lib.utils import str2bool

def get_parser(subparser : _SubParsersAction):
  _, _, tool_name = __name__.split(".")
  module_parser = subparser.add_parser(tool_name, help="Anonymizes a file or Directory")
  module_parser.add_argument("DicomPath", type=Path, help="Path to directory or dicomFile")
  module_parser.add_argument(
    '--keepuids', type=str2bool, nargs='?', const=False, default=False,
      help="toggle to retain SOPInstanceID StudyUID and SeriesUID")
  module_parser.add_argument('--key', type=Path, help="Path to key file for anonymization,this files will contain Personal information!")
  module_parser.add_argument('--pidpf', type=str, default=_PPrefix, help="Prefix for PatientID, so anonymized PatientID will be <pidpf>XXXX where X is the patient number")
  module_parser.add_argument('--pnpf', type=str, default=BASE_ANONYMIZED_PATIENT_NAME, help="Prefix for PatientName, so anonymized PatientName will be <pnpf>XXXX where X is the patient number")
  module_parser.add_argument('--sid', type=str, default="", help="Overwrites the StudyID with <sid>XXXX where X is the patient number")
  module_parser.add_argument('--overwrite', type=str2bool, nargs='?', const=False, default=False,
      help="Delete the directory / file at the destination")

def entry_func(args : Namespace):
  # This is first to find the DicomPath to fail fast.
  if args.DicomPath.is_file():
    target = args.DicomPath.parent / ("anon_" + args.DicomPath.name)
  elif args.DicomPath.is_dir():
    target = args.DicomPath.parent / ("anon_" + args.DicomPath.name)
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


  identityMapping = IdentityMapping()
  identityMapping.fill_from_DicomTree(
    tree,
    patient_prefix=args.pidpf,
    change_UIDs=not args.keepuids
  )

  tree.map(anonymize_dicom_tree(
    identityMapping,
    PatientName=args.pnpf,
    StudyID=args.sid
  ), identityMapping)

  tree.save_tree(target)
