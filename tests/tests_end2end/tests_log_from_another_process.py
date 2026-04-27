"""Tests that we can log from another process"""

# Python standard library
from logging import getLogger
import os
from unittest.mock import patch

# Dicomnode modules
from dicomnode.config import config_from_raw, DicomnodeConfigRaw
from dicomnode.constants import DICOMNODE_PROCESS_LOGGER, DICOMNODE_LOGGER_NAME
from dicomnode.data_structures.image_tree import DicomTree
from dicomnode.dicom.dimse import Address,send_images
from dicomnode.server.input import AbstractInput
from dicomnode.server.input_container import InputContainer
from dicomnode.server.nodes import AbstractPipeline
from dicomnode.server.output import PipelineOutput, NoOutput
from dicomnode.server.processor import AbstractProcessor

# Test
from tests.helpers import config
from tests.helpers import generate_numpy_datasets, personify
from tests.helpers.dicomnode_test_case import DicomnodeTestCase

port = 42134
ae_title = "PIPELINE"
address = Address('127.0.0.1', port, ae_title)

class Input(AbstractInput):
  required_tags = [0x0008_0018]

  def validate(self) -> bool:
    return True

class Processor(AbstractProcessor):
  def process(self, input_container: InputContainer) -> PipelineOutput:
    self.logger.info(f"Hello from {os.getpid()}")
    self.logger.info(f"my handlers are {self.logger.handlers}")


    print("Should be a log message")

    return NoOutput()

class Pipeline(AbstractPipeline):
  ae_title = ae_title
  port = port
  input = {
    'INPUT' : Input
  }

  Processor = Processor

class LogFromAnotherProcess(DicomnodeTestCase):
  def test_end2end_log_from_another_process(self):

    with self.assertLogs(DICOMNODE_LOGGER_NAME) as captured_logs:
      with self.assertLogs(DICOMNODE_PROCESS_LOGGER) as captured_process_logs:
        with patch('dicomnode.lib.logging.set_logger'):
          node = Pipeline(config_from_raw(DicomnodeConfigRaw(
            PROCESSING_DIRECTORY=self._testMethodName
          )))

          node.open(blocking=False)

          images_1 = DicomTree(generate_numpy_datasets(10, PatientID = "1502799995"))

          images_1.map(personify(
            tags=[
              (0x00100010, "PN", "Odd Haugen Test"),
              (0x00100040, "CS", "M")
            ]
          ))

          send_images("TESTCASE", address, images_1)

          node.close()

    self.assertRegexIn("Hello from", captured_process_logs.output)
    self.assertRegexIn("Process has handled 1502799995", captured_process_logs.output)



    if config.PRINT_LOGS:
      from pprint import pp

      pp(captured_logs.output)
      pp(captured_process_logs.output)
