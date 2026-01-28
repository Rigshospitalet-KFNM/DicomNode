from pathlib import Path

from pydicom import Dataset
from pydicom.uid import CTImageStorage

from numpy import ndarray

import logging

from os import environ

from dicomnode.dicom.dimse import Address
from dicomnode.dicom.dicom_factory import DicomFactory
from dicomnode.dicom.blueprints import ct_image_blueprint
from dicomnode.dicom.sop_mapping import CTImageStorage_required_tags
from dicomnode.server.processor import AbstractProcessor
from dicomnode.server.grinders import NumpyGrinder
from dicomnode.server.input import AbstractInput
from dicomnode.server.nodes import AbstractPipeline
from dicomnode.server.output import NoOutput, PipelineOutput, DicomOutput
from dicomnode.server.pipeline_tree import InputContainer


INPUT_KW = "CT_IMAGE"

ENDPOINT = Address('1.2.3.4', 104, "ENDPOINT_AE")
AE_TITLE = "PLUSONE"

class CTInput(AbstractInput):
  required_tags = CTImageStorage_required_tags

  image_grinder = NumpyGrinder()

  def validate(self):
    return self.images > 150

class PlusOnePipeline(AbstractPipeline):
  ae_title: str = AE_TITLE
  port: int = 1337
  ip: str = '0.0.0.0'

  disable_pynetdicom_logger=True
  log_level: int = logging.INFO
  log_output = Path("log.log")



  input = {
    INPUT_KW : CTInput
  }

  class Processor(AbstractProcessor):
    def process(self, input_container: InputContainer) -> PipelineOutput:
      data: ndarray = input_container[INPUT_KW] # type: ignore

      data += 1

      # Conversion back to dicom
      series = DicomFactory().build_series(data,
                                  ct_image_blueprint,
                                  input_container.datasets[INPUT_KW])

      # Producing Pipeline Output
      return DicomOutput([(ENDPOINT, series)], AE_TITLE)
