# This example is borked right now!

import numpy
import logging
import os

from pathlib import Path

from dicomnode.server.grinders import NumpyGrinder
from dicomnode.dicom.dicom_factory import Blueprint, StaticElement, DicomFactory
from dicomnode.dicom.blueprints import SOP_common_blueprint, general_series_blueprint, image_plane_blueprint
from dicomnode.server.nodes import AbstractPipeline
from dicomnode.server.input import DynamicInput
from dicomnode.server.pipeline_tree import InputContainer
from dicomnode.server.output import PipelineOutput, FileOutput, NoOutput
from dicomnode.server.processor import AbstractProcessor

from typing import Dict, Any


DEFAULT_PATH = "/tmp/"
OUTPUT_PATH = Path(os.environ.get("AVERAGE_NODE_OUTPUT_PATH", default=DEFAULT_PATH))

INPUT_KW = "series"
factory = DicomFactory()

blueprint: Blueprint = SOP_common_blueprint \
  + image_plane_blueprint \
  + general_series_blueprint

blueprint.add_virtual_element(StaticElement(0x0008103E,'LO', "Averaged Image"))

class SeriesInputs(DynamicInput):
  image_grinder = NumpyGrinder()
  required_tags = blueprint.get_required_tags()

  def validate(self) -> bool:
    if len(self.data) == 0:
      return False
    lengths = set()
    for leaf in self.data.values():
      lengths.add(len(leaf))
    return len(lengths) == 1 # this checks that all leafs are the same length

class AveragingPipeline(AbstractPipeline):
  header_blueprint = blueprint
  dicom_factory = factory

  ae_title = "AVERAGENODE"
  ip = '0.0.0.0'
  port = 1337

  log_level = logging.DEBUG
  disable_pynetdicom_logger = True

  input = {
    INPUT_KW : SeriesInputs
  }

  class Processor(AbstractProcessor):
    def process(self, input_container: InputContainer):
      studies = numpy.array(input_container[INPUT_KW])
      result = studies.mean(axis=0)

      return NoOutput()

if __name__ == "__main__":
  pipe = AveragingPipeline()
  pipe.open()
