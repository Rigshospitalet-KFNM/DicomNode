from pathlib import Path

from pydicom import Dataset
from pydicom.uid import CTImageStorage

from numpy import ndarray

import logging


from os import environ

from dicomnode.lib.dimse import Address
from dicomnode.lib.numpy_factory import NumpyFactory, CTImageStorage_NumpyBlueprint
from dicomnode.server.grinders import NumpyGrinder
from dicomnode.lib.sop_mapping import CTImageStorage_required_tags

from dicomnode.server.input import AbstractInput
from dicomnode.server.nodes import AbstractPipeline
from dicomnode.server.output import NoOutput, PipelineOutput, DicomOutput
from dicomnode.server.pipeline_tree import InputContainer


INPUT_KW = "CT_IMAGE"

class CTInput(AbstractInput):
  required_tags = CTImageStorage_required_tags

  image_grinder = NumpyGrinder()

  def validate(self):
    return self.images > 150

class PlusOnePipeline(AbstractPipeline):
  log_path: str = "log.log"
  ae_title: str = "PLUSONE"
  port: int = 1337
  ip: str = '0.0.0.0'
  disable_pynetdicom_logger=True
  log_level: int = logging.INFO
  header_blueprint = CTImageStorage_NumpyBlueprint

  endpoint = Address('1.2.3.4', 104, "ENDPOINT_AE")

  dicom_factory: NumpyFactory = NumpyFactory()

  input = {
    INPUT_KW : CTInput
  }

  def process(self, input_data: InputContainer) -> PipelineOutput:
    data: ndarray = input_data[INPUT_KW] # type: ignore
    if input_data.header is None:
      self.logger.critical("Header is missing!")
      raise Exception
    # Data processing
    data += 1

    # Conversion back to dicom
    series = self.dicom_factory.build_from_header(input_data.header, data)

    # Producing Pipeline Output
    out = DicomOutput([(self.endpoint, series)], self.ae_title)

    return out

