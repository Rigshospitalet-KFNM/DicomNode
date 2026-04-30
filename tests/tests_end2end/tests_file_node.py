# Python3 Standard library
import logging
import os
from pathlib import Path
from random import randint
from unittest import TestCase
from time import sleep

# Third Party Packages

# Dicomnode Packages
from dicomnode.constants import DICOMNODE_LOGGER_NAME, DICOMNODE_PROCESS_LOGGER
from dicomnode.data_structures.image_tree import DicomTree
from dicomnode.dicom.dimse import Address, send_images
from dicomnode.server.nodes import AbstractPipeline
from dicomnode.server.input_container import InputContainer
from dicomnode.server.output import NoOutput, PipelineOutput
from dicomnode.server.processor import AbstractProcessor

# Testing Helper
from tests.helpers import TESTING_TEMPORARY_DIRECTORY, bench,\
  generate_numpy_datasets, personify, clear_logger
from tests.helpers.inputs import ValidatingInput
from tests.helpers.dicomnode_test_case import DicomnodeTestCase

DICOM_STORAGE_PATH = Path(f"{TESTING_TEMPORARY_DIRECTORY}/file_storage")
PROCESSING_DIRECTORY = Path(f"{TESTING_TEMPORARY_DIRECTORY}/working_directory")

ENDPOINT_PORT = 50000
ENDPOINT_AE = "ENDPOINT_AT"

TEST_CPR = "1502799995"
INPUT_KW = "test_input"
HISTORIC_KW = "historic_input"
TEST_AE_TITLE = "TEST_AE"
SENDER_AE_TITLE = "SENDER_AE"


class StallingFileStorageNode(AbstractPipeline):
  ae_title = TEST_AE_TITLE
  input = {INPUT_KW : ValidatingInput }
  require_calling_aet = [SENDER_AE_TITLE]
  log_output = "log.log"
  log_level: int = logging.DEBUG
  disable_pynetdicom_logger: bool = True
  root_data_directory = DICOM_STORAGE_PATH
  processing_directory = PROCESSING_DIRECTORY

  class Processor(AbstractProcessor):
    def process(self, input_container: InputContainer) -> PipelineOutput:
      log_message =  f"process is called at cwd: {os.getcwd()}"
      self.logger.info(log_message)
      return NoOutput()


class StallingFileStorageTestCase(DicomnodeTestCase):
  def test_check_log_file_is_created(self):
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    DICOM_STORAGE_PATH.mkdir(parents=True, exist_ok=True)
    self.node = StallingFileStorageNode()
    self.test_port = randint(1025,65535)
    self.node.port = self.test_port
    self.node.open(blocking=False)



    address = Address('localhost', self.test_port, TEST_AE_TITLE)
    images_1 = DicomTree(generate_numpy_datasets(10, PatientID = "1502799995", Cols=10, Rows=10))

    logger = logging.getLogger(DICOMNODE_LOGGER_NAME)

    images_1.map(personify(
      tags=[
        (0x00100010, "PN", "Odd Haugen Test"),
        (0x00100040, "CS", "M")
      ]
    ))

    send_images(SENDER_AE_TITLE, address, images_1, None)

    self.node.close()

    log_file_path = Path("/tmp/pipeline_tests/log.log")
    self.assertTrue(log_file_path.exists())

    log_text = log_file_path.read_text()
    self.assertIn('process is called at cwd: /tmp/pipeline_tests/working_directory/1502799995', log_text)

    clear_logger(DICOMNODE_LOGGER_NAME)
    clear_logger(DICOMNODE_PROCESS_LOGGER)
