
from email.policy import default
from pathlib import Path
from shutil import rmtree


from argparse import _SubParsersAction, Namespace
from dicomnode.lib.anonymization import anonymize_dataset, BASE_ANONYMIZED_PATIENT_NAME
from dicomnode.lib.io import discover_dicom_files
from dicomnode.lib.studyTree import DicomTree, IdentityMapping, _PPrefix


from dicomnode.lib.utils import str2bool

def get_parser(subparser : _SubParsersAction):
  _, _, tool_name = __name__.split(".")
  module_parser = subparser.add_parser(tool_name, help="Anonymizes a file or Directory")
  module_parser.add_argument("DicomPath", type=Path, help="Path to directory or dicomFile")
  module_parser.add_argument(
    '--keepuids', type=str2bool, nargs='?', const=False, default=False,
      help="toggle to retain SOPInstanceID StudyUID and SeriesUID")
  module_parser.add_argument('--pidpf', type=str, default=_PPrefix)
  module_parser.add_argument('--pnpf', type=str, default=BASE_ANONYMIZED_PATIENT_NAME)
  module_parser.add_argument('--sid', type=str, default="")
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
  discover_dicom_files(args.DicomPath, tree)

  identityMapping = IdentityMapping()
  identityMapping.fill_from_DicomTree(
    tree,
    patient_prefix=args.pidpf,
    change_UIDs=not args.keepuids
  )

  tree.apply_mapping(anonymize_dataset(
    identityMapping,
    PatientName=args.pnpf,
    StudyID=args.sid
  ), identityMapping)

  tree.save_tree(target)
