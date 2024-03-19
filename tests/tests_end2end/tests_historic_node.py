"""This is a test case of a historic input"""

# Python3 standard library
from random import randint
import logging
from time import sleep
from unittest import TestCase, skip

# Third party Packages

# Dicomnode Packages
from dicomnode.dicom.dimse import Address, send_image
from dicomnode.server.nodes import AbstractPipeline
from dicomnode.server.pipeline_tree import InputContainer
from dicomnode.server.output import PipelineOutput, NoOutput

# Testing Packages
from tests.helpers import get_test_ae
from tests.helpers.storage_endpoint import TestStorageEndpoint, ENDPOINT_PORT
from tests.helpers.inputs import TestHistoricInput, NeverValidatingInput

# Constants
TEST_AE_TITLE = "TEST_AE"
SENDER_AE_TITLE = "SENDER_AE"

INPUT_KW = "input"
HISTORIC_KW = "historic"

class HistoricPipeline(AbstractPipeline):
  ae_title = TEST_AE_TITLE
  input = {
    INPUT_KW : NeverValidatingInput,
    HISTORIC_KW : TestHistoricInput
   }
  require_calling_aet = [SENDER_AE_TITLE, "DUMMY"]
  log_level: int = logging.CRITICAL
  disable_pynetdicom_logger: bool = False
  processing_directory = None
  log_output = None

  def process(self, input_container: InputContainer) -> PipelineOutput:
    return NoOutput()

class HistoricTestCase(TestCase):
  def setUp(self) -> None:
    self.node = HistoricPipeline()
    self.test_port = randint(1025,65535)
    self.node.port = self.test_port
    self.node.open(blocking=False)

  def tearDown(self) -> None:
    self.node.close()

  @skip # Yes it's broken
  def test_create_and_send(self):
    address = Address("localhost", self.test_port, TEST_AE_TITLE)
    endpoint = get_test_ae(ENDPOINT_PORT, self.test_port, self.node.logger)

    with self.assertLogs(self.node.logger, logging.DEBUG) as cm:
      response = send_image(SENDER_AE_TITLE, address, DEFAULT_DATASET)
      sleep(0.25) # wait for all the threads to be done
    endpoint.shutdown()
