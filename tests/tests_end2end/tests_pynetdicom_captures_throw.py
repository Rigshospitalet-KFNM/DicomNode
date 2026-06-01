"""This test case is to show that pynetdicom captures the error logs of things"""
# Also maybe I should create a single file for end2end logging...

# Python Standard library
import logging
from io import StringIO
from random import randint

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
from dicomnode.server.nodes import AbstractPipeline
from dicomnode.server.output import PipelineOutput
from dicomnode.server.processor import AbstractProcessor

# Test helpers
from tests.helpers import generate_numpy_datasets, clear_logger
from tests.helpers.dicomnode_test_case import DicomnodeTestCase


class Input(AbstractInput):
  def validate(self) -> bool:
    return True

class PynetdicomNodeLogging(DicomnodeTestCase):
  def test_leaked_exceptions_are_captured_in_pynetdicom_logs(self):
    output = StringIO()
    ae_title_ = "IMINDANGER"
    port_ = randint(1050,45000)

    class Node(AbstractPipeline):
      input = {
        "PROBLEM" : Input
      }

      pynetdicom_logger_config = LoggerConfig(
        log_output=output,
        log_level=logging.ERROR,
        propagate=False
      )

      def _handle_c_store(self, event: Event) -> int:
        raise Exception("I'm the Problem!")

      ae_title = ae_title_
      port = port_

      class Processor(AbstractProcessor):
        def process(self, input_container: InputContainer) -> PipelineOutput:
          return super().process(input_container)

    node = Node()

    with self.assertLogs(DICOMNODE_LOGGER_NAME):
      with self.assertLogs("pynetdicom.service_class", logging.ERROR) as captured_logs:
        with node.open_cm():
          address = Address('localhost', port_, ae_title_)
          with self.assertRaises(CouldNotCompleteDIMSEMessage):
            send_images("SOAMI", address, list(generate_numpy_datasets(2, Cols=5, Rows=5)))

    clear_logger(DICOMNODE_LOGGER_NAME)

    self.assertRegexIn("I'm the Problem!", captured_logs.output)
