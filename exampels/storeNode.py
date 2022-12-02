from pathlib import Path

from pydicom import Dataset

from os import environ

from dicomnode.server.input import AbstractInput
from dicomnode.server.nodes import AbstractPipeline
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

  input: Dict[str, type] = {
    INPUT_ARG : DicomObjectInput
  }

  archive_path: Path = Path(ARCHIVE_PATH)

  def storeDataset(self, dataset) -> None:
    dataset_path: Path = self.archive_path / dataset.StudyInstanceUID.name / dataset.SeriesInstanceUID.name / (dataset.SOPInstanceUID.name + '.dcm')
    save_dicom(dataset_path,  dataset)
    return None

  def process(self, input_data: InputContainer) -> Iterable[Dataset]:
    for dataset in input_data[INPUT_ARG]:
      self.storeDataset(dataset)
    return []

  def post_init(self, _: bool) -> None:
    self.archive_path.mkdir(exist_ok=True)


if __name__ == "__main__":
  storeNode()