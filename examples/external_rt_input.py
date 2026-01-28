"""These input are made in an example where you have some external service that
can produce a RT struct

"""
# Python standard library
from os import environ

# Third party Modules
from pydicom import Dataset
from pydicom.uid import RTImageStorage, CTImageStorage

# Dicomnode Modules
from dicomnode.lib.exceptions import CouldNotCompleteDIMSEMessage
from dicomnode.dicom.dimse import Address, send_images
from dicomnode.lib.validators import OptionsValidator
from dicomnode.server.grinders import RTStructGrinder
from dicomnode.server.input import AbstractInput

ENV_VAR_address_NAME = "PIPELINE_RT_ADDRESS"
ENV_VAR_port_NAME = "PIPELINE_RT_PORT"
ENV_VAR_ae_title_NAME = "PIPELINE_RT_AE_TITLE"

SERVICE_ADDRESS = environ.get(ENV_VAR_address_NAME)
SERVICE_PORT = environ.get(ENV_VAR_port_NAME)
SERVICE_AE_TITLE = environ.get(ENV_VAR_ae_title_NAME)

class RTConsumer(AbstractInput):
  image_grinder = RTStructGrinder()

  required_values = {
    0x0008_0016 : OptionsValidator([
      RTImageStorage, CTImageStorage
    ])
  }

  def __init__(self, options: AbstractInput.Options = AbstractInput.Options()):
    super().__init__(options)
    self.has_rt_struct = False
    self.has_send_datasets = False

  def add_image(self, dicom: Dataset) -> int:
    return_value = super().add_image(dicom)
    if return_value != 0 and dicom.SOPClassUID == RTImageStorage:
      self.has_rt_struct = True
    return return_value

  def validate(self) -> bool:
    ready = 1 < self.images and self.has_rt_struct
    if not ready and 0 < self.images and self.has_send_datasets:
      self.has_send_datasets = True

      if SERVICE_ADDRESS is not None and\
         SERVICE_PORT is not None and\
         SERVICE_AE_TITLE is not None and\
         self.options.ae_title is not None:

        try:
          send_images(self.options.ae_title,
                      Address(SERVICE_ADDRESS,
                              int(SERVICE_PORT),
                              SERVICE_AE_TITLE), self)
        except CouldNotCompleteDIMSEMessage:
          error_message = "Could send images with parameters:\n"\
                         f"IP:   {SERVICE_ADDRESS}\n"\
                         f"port: {SERVICE_PORT}\n"\
                         f"scp: {SERVICE_AE_TITLE}\n"\
                         f"scu: {self.options.ae_title}\n"
          self.logger.error(error_message)
      else:
        error_message = f"This Input is incorrectly configured the following "\
          "values should not be None:\n"\
          f"Environment Variable: {ENV_VAR_address_NAME} - {SERVICE_ADDRESS}\n"\
          f"Environment Variable: {ENV_VAR_port_NAME} - {SERVICE_PORT}\n"\
          f"Environment Variable: {ENV_VAR_ae_title_NAME} - {SERVICE_AE_TITLE}\n"\
          f"AE SCU: {self.options.ae_title}"
        self.logger.error(error_message)

    return ready
