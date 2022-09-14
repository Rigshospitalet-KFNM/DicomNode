"""This module contains basic DIMSE messages
  """

__author__ = "Christoffer Vilstrup Jensen"

from pydicom import Dataset
from pynetdicom.ae import ApplicationEntity

from dicomnode.lib.exceptions import CouldNotCompleteDIMSEMessage

def send_c_store(ip : str, port : int, SCP_AE : str, SCU_AE : str, dicom_image : Dataset ) -> Dataset:
  ae = ApplicationEntity(SCU_AE)

  ae.add_requested_context(dicom_image.SOPClassUID, dicom_image.TransferSyntaxUID)
  assoc = ae.associate(ip, port, SCP_AE)
  if assoc.is_established:
    response = assoc.send_c_store(dicom_image)
    assoc.release()
    return response
  else:
    print(f"Could not connect to the SCP with the following inputs:\nIP: {ip}\nPort: {port}\nSCP AE: {SCP_AE}\n SCU AE:{SCU_AE}")
    raise CouldNotCompleteDIMSEMessage("Could not connect")

