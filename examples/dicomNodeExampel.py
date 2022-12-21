from pydicom import Dataset

from dicomnode.server.input import AbstractInput
from dicomnode.server.pipelineTree import InputContainer
from dicomnode.server.nodes import AbstractPipeline

from typing import Iterable

class storeNode(AbstractPipeline):
  log_path = "playground/exampel/logs/log.log"
  ae_title = "EXAMPLE"
  port     = 11112

  def process(self, input_data: InputContainer) -> Iterable[Dataset]:
    return super().process(input_data)

if __name__ == "__main__":
  storeNode()