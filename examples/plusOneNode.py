from pathlib import Path

from pydicom import Dataset
from pydicom.uid import CTImageStorage

from numpy import ndarray

import logging

from os import environ

from dicomnode.dicom.dimse import Address
from dicomnode.dicom.dicom_factory import DicomFactory
from dicomnode.dicom.blueprints import ct_image_blueprint
from dicomnode.server.grinders import NumpyGrinder
from dicomnode.dicom.sop_mapping import CTImageStorage_required_tags

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
  ae_title: str = "PLUSONE"
  port: int = 1337
  ip: str = '0.0.0.0'

  disable_pynetdicom_logger=True
  log_level: int = logging.INFO
  log_output: str = "log.log"

  endpoint = Address('1.2.3.4', 104, "ENDPOINT_AE")

  input = {
    INPUT_KW : CTInput
  }

  def process(self, input_data: InputContainer) -> PipelineOutput:
    data: ndarray = input_data[INPUT_KW] # type: ignore

    data += 1

    # Conversion back to dicom
    series = DicomFactory().build_series(data,
                                ct_image_blueprint,
                                input_data.datasets[INPUT_KW])

    # Producing Pipeline Output
    out = DicomOutput([(self.endpoint, series)], self.ae_title)

    return out

