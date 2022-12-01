"""Library methods for manipulation of pydicom.dataset objects
"""

from dicomnode.constants import DICOMNODE_IMPLEMENTATION_UID, DICOMNODE_IMPLEMENTATION_NAME, DICOMNODE_VERSION

from dicomnode.lib.exceptions import InvalidDataset

from pydicom import Dataset
from pydicom.uid import UID, generate_uid, ImplicitVRLittleEndian, ExplicitVRBigEndian, ExplicitVRLittleEndian

def gen_uid() -> UID:
  return generate_uid(prefix=DICOMNODE_IMPLEMENTATION_UID + '.')

def make_meta(dicom: Dataset) -> None:
  """Similar to fix_meta_info method, however UID are generated with dicomnodes prefix instead

  Args:
      dicom (Dataset): dicom dataset to be updated
  Raises:
      InvalidDataset: If meta header cannot be generated or is Transfer syntax is not supported
  """
  if dicom.is_little_endian is None:
    dicom.is_little_endian = True
  if dicom.is_implicit_VR is None:
    dicom.is_implicit_VR = True
  if not 0x00080016 in dicom:
    raise InvalidDataset("Cannot create meta header without SOPClassUID")
  if not 0x00080018 in dicom:
    dicom.SOPInstanceUID = gen_uid()

  dicom.ensure_file_meta()

  dicom.file_meta.FileMetaInformationVersion = b'\x00\x01'
  dicom.file_meta.ImplementationClassUID = DICOMNODE_IMPLEMENTATION_UID
  dicom.file_meta.ImplementationVersionName = f"{DICOMNODE_IMPLEMENTATION_NAME} {DICOMNODE_VERSION}"
  dicom.file_meta.MediaStorageSOPClassUID = dicom.SOPClassUID
  dicom.file_meta.MediaStorageSOPInstanceUID = dicom.SOPInstanceUID

  if dicom.is_little_endian and dicom.is_implicit_VR:
    dicom.file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
  elif dicom.is_little_endian and not dicom.is_implicit_VR:
    dicom.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
  elif not dicom.is_little_endian and not dicom.is_implicit_VR:
    dicom.file_meta.TransferSyntaxUID = ExplicitVRBigEndian
  else:
    raise InvalidDataset("Implicit VR Big Endian is not a "
                         "supported Transfer Syntax.")





