from unittest import TestCase
from random import randint
from pydicom import Dataset
from time import sleep
import logging


from dicomnode.lib.dicom import gen_uid, make_meta
from dicomnode.lib.dimse import Address, send_image
from dicomnode.lib.exceptions import CouldNotCompleteDIMSEMessage

from dicomnode.server.input import AbstractInput
from dicomnode.server.nodes import AbstractPipeline
from pydicom.uid import RawDataStorage, ImplicitVRLittleEndian

from typing import List, Dict, Any, Iterator

TEST_AE_TITLE = "TEST_AE"
SENDER_AE = "SENDER_AE"

DEFAULT_DATASET = Dataset()
DEFAULT_DATASET.SOPClassUID = RawDataStorage
make_meta(DEFAULT_DATASET)


DEFAULT_DATASET.PatientID = "1502799995"

class TestInput(AbstractInput):
  required_tags: List[int] = [0x00080018]

  def validate(self):
    return True


class NodeTestImplementation(AbstractPipeline):
  ae_title = TEST_AE_TITLE
  input = {'test_input' : TestInput }
  require_calling_aet = [SENDER_AE]
  log_level: int = logging.CRITICAL
  disable_pynetdicom_logger: bool = True

  def process(self, InputData: Dict[str, Any]) -> Iterator[Dataset]:
    self.logger.info("process is called")
    return []


class ServerNodesTestCase(TestCase):
  def setUp(self):
    self.node = NodeTestImplementation(start=False)
    self.test_port = randint(1025,65535)
    self.node.port = self.test_port
    self.node.open(blocking=False)

  def tearDown(self) -> None:
    self.node.close()


  def test_send_C_store_success(self):
    address = Address('localhost', self.test_port, TEST_AE_TITLE)
    response = send_image(SENDER_AE, address, DEFAULT_DATASET)
    self.assertEqual(response.Status, 0x0000)


  def test_reject_connection(self):
    address = Address('localhost', self.test_port, TEST_AE_TITLE)
    self.assertRaises(CouldNotCompleteDIMSEMessage,send_image,"NOT_SENDER_AE", address, DEFAULT_DATASET)

