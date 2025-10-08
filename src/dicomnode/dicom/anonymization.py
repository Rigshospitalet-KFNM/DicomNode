"""Module docs"""

# Python standard library
from typing import Callable, Optional

# Third party packages
from pydicom import Dataset
from pydicom.valuerep import VR
from pydicom.tag import BaseTag

# Dicom node packages
from dicomnode.data_structures import image_tree

BASE_ANONYMIZED_PATIENT_NAME = "Anonymized_PatientName"

def anonymize_dataset(ds: Dataset, PatientName="Anon", PatientNumber="0000"):
  if hasattr(ds, 'file_meta'):
    anonymize_dataset(ds.file_meta)

  for dataElement in ds:
    vr = dataElement.VR # The VR of a data set is either a pydicom.valuerep.VR or str
    if not isinstance(vr, VR):
      vr = VR(vr)
      dataElement.VR = vr
    if dataElement.tag == BaseTag(0x00100010): # PatientName tag value
      dataElement.value  = f"{PatientName}_{PatientNumber}"
    if dataElement.tag == BaseTag(0x00100020): # PatientID tag value
      dataElement.value  = f"{PatientNumber}"
    elif vr == VR.PN:
      dataElement.value = "Anon_" + dataElement.name
    elif vr == VR.SQ:
      for seq in dataElement:
        anonymize_dataset(seq)


def anonymize_dicom_tree(
  UIDMapping : image_tree.IdentityMapping,
  PatientName : str = BASE_ANONYMIZED_PATIENT_NAME,
  StudyID : Optional[str] = "Study"
  ) -> Callable[[Dataset], None]:
  """Creates a function

  Args:
      UIDMapping (IdentityMapping): _description_
      PatientName (str, optional): _description_. Defaults to BASE_ANONYMIZED_PATIENT_NAME.
      StudyID (Optional[str], optional): _description_. Defaults to "Study".

  Returns:
      Callable[[Dataset], None]: _description_
  """
  def retFunc(dataset: Dataset) -> None:
    newPatientID = UIDMapping.PatientMapping[dataset.PatientID]
    dataset.PatientID = newPatientID
    PatientNumber = newPatientID[-UIDMapping.prefix_size:]
    if StudyID:
      dataset.StudyID = f"{StudyID}_{PatientNumber}"

    def anon_ds(ds: Dataset):
      if hasattr(ds, 'file_meta'):
        anon_ds(ds.file_meta)

      for dataElement in ds:
        vr = dataElement.VR # The VR of a data set is either a pydicom.valuerep.VR or str
        if not isinstance(vr, VR): # The VR might be very wierd
          vr = VR(vr)
          dataElement.VR = vr
        if dataElement.tag == BaseTag(0x00100010): # PatientName tag value
          dataElement.value  = f"{PatientName}_{PatientNumber}"
        elif vr == VR.PN:
          dataElement.value = "Anon_" + dataElement.name
        elif vr == VR.SQ:
          for seq in dataElement:
            anon_ds(seq)
        elif vr == VR.UI:
          if newUID := UIDMapping.get_mapping(dataElement.value):
            dataElement.value = newUID
    anon_ds(dataset)

  return retFunc
