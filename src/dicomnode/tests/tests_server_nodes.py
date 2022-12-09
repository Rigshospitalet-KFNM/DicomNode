from unittest import TestCase
from random import randint
from pydicom import Dataset
from time import sleep
import logging

from sys import getrefcount

from dicomnode.lib.dicom import gen_uid, make_meta
from dicomnode.lib.dimse import Address, send_image
from dicomnode.lib.exceptions import CouldNotCompleteDIMSEMessage

from dicomnode.server.input import AbstractInput
from dicomnode.server.nodes import AbstractPipeline
from pydicom.uid import RawDataStorage, ImplicitVRLittleEndian

from typing import List, Dict, Any, Iterable

TEST_AE_TITLE = "TEST_AE"
SENDER_AE = "SENDER_AE"

DEFAULT_DATASET = Dataset()
DEFAULT_DATASET.SOPClassUID = RawDataStorage
make_meta(DEFAULT_DATASET)
DATASET_SOPINSTANCEUID = DEFAULT_DATASET.SOPInstanceUID.name


TEST_CPR = "1502799995"

INPUT_KW = "test_input"

DEFAULT_DATASET.PatientID = TEST_CPR

class TestInput(AbstractInput):
  required_tags: List[int] = [0x00080018]

  def validate(self):
    return True


class TestNode(AbstractPipeline):
  ae_title = TEST_AE_TITLE
  input = {INPUT_KW : TestInput }
  require_calling_aet = [SENDER_AE]
  log_level: int = logging.CRITICAL
  disable_pynetdicom_logger: bool = True

  def process(self, InputData: Dict[str, Any]) -> Iterable[Dataset]:
    self.logger.info("process is called")
    return []


class TestNodeTestCase(TestCase):
  def setUp(self):
    self.node = TestNode(start=False)
    self.test_port = randint(1025,65535)
    self.node.port = self.test_port
    self.node.open(blocking=False)

  def tearDown(self) -> None:
    self.node.close()

  def test_send_C_store_success(self):
    address = Address('localhost', self.test_port, TEST_AE_TITLE)
    response = send_image(SENDER_AE, address, DEFAULT_DATASET)
    self.assertEqual(response.Status, 0x0000)
    # Okay This is mostly to ensure lazyness
    # See the advanced docs guide for details
    self.assertEqual(getrefcount(self.node._AbstractPipeline__data_state.data[TEST_CPR].data[INPUT_KW].data[DATASET_SOPINSTANCEUID]), 2) # type: ignore

  def test_reject_connection(self):
    address = Address('localhost', self.test_port, TEST_AE_TITLE)
    self.assertRaises(CouldNotCompleteDIMSEMessage,send_image,"NOT_SENDER_AE", address, DEFAULT_DATASET)


class FaultyNode(AbstractPipeline):
  ae_title = TEST_AE_TITLE
  input = {INPUT_KW : TestInput }
  require_calling_aet = [SENDER_AE]
  log_level: int = logging.CRITICAL
  disable_pynetdicom_logger: bool = True

  def process(self, InputData: Dict[str, Any]) -> Iterable[Dataset]:
    raise Exception

class FaultyNodeTestCase(TestCase):
  def setUp(self):
    self.node = FaultyNode(start=False)
    self.test_port = randint(1025,65535)
    self.node.port = self.test_port
    self.node.open(blocking=False)

  def tearDown(self) -> None:
    self.node.close()

  def test_faulty_process(self):
    address = Address('localhost', self.test_port, TEST_AE_TITLE)
    response = send_image(SENDER_AE, address, DEFAULT_DATASET)
    self.assertEqual(response.Status, 0x0000)
