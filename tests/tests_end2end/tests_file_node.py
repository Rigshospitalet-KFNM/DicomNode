# Python3 Standard library
import logging
import os
from pathlib import Path
from random import randint
from unittest import TestCase
from time import sleep

# Third Party Packages

# Dicomnode Packages
from dicomnode.data_structures.image_tree import DicomTree
from dicomnode.dicom.dimse import Address, send_images_thread
from dicomnode.server.nodes import AbstractPipeline
from dicomnode.server.pipeline_tree import InputContainer
from dicomnode.server.output import NoOutput, PipelineOutput

# Testing Helper
from tests.helpers import TESTING_TEMPORARY_DIRECTORY, bench,\
  generate_numpy_datasets, personify
from tests.helpers.inputs import NeverValidatingInput

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
  input = {INPUT_KW : NeverValidatingInput }
  require_calling_aet = [SENDER_AE_TITLE]
  log_output = None
  log_level: int = logging.DEBUG
  disable_pynetdicom_logger: bool = True
  root_data_directory = DICOM_STORAGE_PATH
  processing_directory = PROCESSING_DIRECTORY

  def process(self, input_data: InputContainer) -> PipelineOutput:
    log_message =  f"process is called at cwd: {os.getcwd()}"
    self.logger.info(log_message)
    return NoOutput()


class StallingFileStorageTestCase(TestCase):
  def setUp(self):
    DICOM_STORAGE_PATH.mkdir(parents=True, exist_ok=True)
    self.node = StallingFileStorageNode()
    self.test_port = randint(1025,65535)
    self.node.port = self.test_port
    self.node.open(blocking=False)

  def tearDown(self) -> None:
    while self.node.dicom_application_entry.active_associations != []:
      sleep(0.005) #pragma: no cover

    #pprint([t for t in threading.enumerate()])
    self.node.close()

  @bench
  def performance_send_fs(self):
    address = Address('localhost', self.test_port, TEST_AE_TITLE)
    images_1 = DicomTree(generate_numpy_datasets(50, PatientID = "1502799995"))

    images_1.map(personify(
      tags=[
        (0x00100010, "PN", "Odd Haugen Test"),
        (0x00100040, "CS", "M")
      ]
    ))

    thread_1 = send_images_thread(SENDER_AE_TITLE, address, images_1, None, False)

    ret_1 = thread_1.join()
    self.assertEqual(ret_1, 0)
    self.assertEqual(self.node.data_state.images,50) # type: ignore