import logging
from pathlib import Path
from os import environ
from typing import Any, Dict, Iterable, List

from pydicom import Dataset

from dicomnode.server.processor import AbstractProcessor
from dicomnode.server.grinders import ListGrinder
from dicomnode.server.input import AbstractInput
from dicomnode.server.nodes import AbstractPipeline
from dicomnode.server.output import FileOutput, PipelineOutput
from dicomnode.server.pipeline_tree import InputContainer
from dicomnode.lib.io import save_dicom


env_name = "STORENODE_ARCHIVE_PATH"
env_default = "/raid/dicom/storenode/"

ARCHIVE_PATH = environ.get(env_name, env_default)

INPUT_ARG: str = "dataset"

class DicomObjectInput(AbstractInput):
  required_tags = [
    0x00080016, # SOPInstanceUID
    0x00100020, # PatientID
    0x0020000D, # StudyInstanceUID
    0x0020000E, # SeriesInstanceUID
  ]

  image_grinder = ListGrinder()

  def validate(self):
    return True

archive_path: Path = Path(ARCHIVE_PATH)

class StoreNode(AbstractPipeline):
  log_output = "log.log"
  ae_title: str = "STORENODE"
  port: int = 1337
  ip: str = '0.0.0.0'
  disable_pynetdicom_logger=True
  log_level: int = logging.INFO

  input: Dict[str, type] = {
    INPUT_ARG : DicomObjectInput
  }

  class Processor(AbstractProcessor):
    def process(self, input_container: InputContainer) -> PipelineOutput:
      return FileOutput([(archive_path, input_container[INPUT_ARG])], saving_function=save_dicom)

  def post_init(self) -> None:
    archive_path.mkdir(exist_ok=True)


if __name__ == "__main__":
  node = StoreNode()
  node.open()
