"""This test case is to show that pynetdicom captures the error logs of things"""
# Also maybe I should create a single file for end2end logging...

# Python Standard library
import logging
from io import StringIO
from random import randint
from unittest.mock import patch


# Third Party modules
from pydicom import Dataset
from pynetdicom.events import Event

# Dicomnode Modules
from dicomnode.constants import DICOMNODE_LOGGER_NAME
from dicomnode.lib.exceptions import CouldNotCompleteDIMSEMessage
from dicomnode.dicom.dimse import Address, send_images
from dicomnode.lib.logging import LoggerConfig
from dicomnode.server.input import AbstractInput
from dicomnode.server.input_container import InputContainer
from dicomnode.server.nodes import AbstractQueuedPipeline
from dicomnode.server.output import PipelineOutput
from dicomnode.server.processor import AbstractProcessor

# Test helpers
from tests.helpers import generate_numpy_datasets, clear_logger
from tests.helpers.dicomnode_test_case import DicomnodeTestCase


class Input(AbstractInput):
  def validate(self) -> bool:
    return True

class PynetdicomNodeLogging(DicomnodeTestCase):
  @patch("dicomnode.lib.logging.set_logger")
  def test_abstract_queued_pipeline_end2end(self, mock):
    output = StringIO()
    ae_title_ = "IMINDANGER"
    port_ = randint(1050,45000)

    class Node(AbstractQueuedPipeline):
      input = {
        "NoProblemHere" : Input
      }

      pynetdicom_logger_config = LoggerConfig(
        log_output=output,
        log_level=logging.ERROR,
        propagate=False
      )

      ae_title = ae_title_
      port = port_

      class Processor(AbstractProcessor):
        def process(self, input_container: InputContainer) -> PipelineOutput:
          self.logger.info("I'm triggered!")
          self.logger.info(f"This is my handlers: {self.logger.handlers}")
          return super().process(input_container)


    with self.assertLogs(DICOMNODE_LOGGER_NAME) as captured_logs:
      node = Node()
      with node.open_cm():
        address = Address('localhost', port_, ae_title_)
        send_images("SOAMI", address, list(generate_numpy_datasets(2, Cols=5, Rows=5)))

        while not node.process_queue.empty():
          pass

    self.assertRegexIn("I'm triggered!", captured_logs.output)


    clear_logger(DICOMNODE_LOGGER_NAME)
