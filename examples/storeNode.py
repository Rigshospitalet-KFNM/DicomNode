from pathlib import Path

from pydicom import Dataset
import logging

from os import environ

from dicomnode.server.input import AbstractInput
from dicomnode.server.nodes import AbstractPipeline
from dicomnode.server.output import FileOutput, PipelineOutput
from dicomnode.server.pipelineTree import InputContainer
from dicomnode.lib.io import save_dicom

from typing import Any, Dict, Iterable, List

env_name = "STORENODE_ARCHIVE_PATH"
env_default = "/raid/dicom/storenode/"

ARCHIVE_PATH = environ.get(env_name, env_default)

INPUT_ARG: str = "dataset"

class DicomObjectInput(AbstractInput):
  required_tags: List[int] = [
    0x00080016, # SOPInstanceUID
    0x0020000D, # StudyInstanceUID
    0x0020000E, # SeriesInstanceUID
  ]

  required_values: Dict[int, Any]

  def validate(self):
    return True


class storeNode(AbstractPipeline):
  log_path: str = "log.log"
  ae_title: str = "STORENODE"
  port: int = 1337
  ip: str = '0.0.0.0'
  disable_pynetdicom_logger=True
  log_level: int = logging.INFO

  input: Dict[str, type] = {
    INPUT_ARG : DicomObjectInput
  }

  archive_path: Path = Path(ARCHIVE_PATH)

  def process(self, input_data: InputContainer) -> PipelineOutput:
    datasets: List[Dataset] = [dataset for dataset in input_data[INPUT_ARG]]

    return FileOutput([(self.archive_path, datasets)])

  def post_init(self, start: bool) -> None:
    self.archive_path.mkdir(exist_ok=True)


if __name__ == "__main__":
  storeNode()
