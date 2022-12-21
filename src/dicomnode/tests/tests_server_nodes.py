from copy import deepcopy
import logging
from random import randint
from pathlib import Path
from sys import getrefcount
from typing import List, Dict, Any, Iterable
from unittest import TestCase

from pydicom import Dataset
from pydicom.uid import RawDataStorage, ImplicitVRLittleEndian

from dicomnode.lib.dicom import gen_uid, make_meta
from dicomnode.lib.dimse import Address, send_image, send_images_thread
from dicomnode.lib.exceptions import CouldNotCompleteDIMSEMessage
from dicomnode.lib.imageTree import DicomTree

from dicomnode.tests.helpers import generate_numpy_datasets, personify, bench

from dicomnode.server.input import AbstractInput
from dicomnode.server.nodes import AbstractPipeline, InputContainer
from dicomnode.server.output import NoOutput, PipelineOutput


TEST_AE_TITLE = "TEST_AE"
SENDER_AE = "SENDER_AE"

DEFAULT_DATASET = Dataset()
DEFAULT_DATASET.SOPClassUID = RawDataStorage
DEFAULT_DATASET.PatientSex = 'M'
make_meta(DEFAULT_DATASET)
DATASET_SOPInstanceUID = DEFAULT_DATASET.SOPInstanceUID.name


TEST_CPR = "1502799995"
INPUT_KW = "test_input"

DEFAULT_DATASET.PatientID = TEST_CPR

class TestInput(AbstractInput):
  required_tags: List[int] = [0x00080018, 0x00100040]

  def validate(self):
    return True


class TestNode(AbstractPipeline):
  ae_title = TEST_AE_TITLE
  input = {INPUT_KW : TestInput }
  require_calling_aet = [SENDER_AE]
  log_level: int = logging.CRITICAL
  disable_pynetdicom_logger: bool = True

  def process(self, InputData: InputContainer) -> PipelineOutput:
    self.logger.info("process is called")
    return NoOutput()


class PipelineTestCase(TestCase):
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
    self.assertEqual(getrefcount(self.node._AbstractPipeline__data_state.data[TEST_CPR].data[INPUT_KW].data[DATASET_SOPInstanceUID]), 2) # type: ignore

  def test_reject_connection(self):
    address = Address('localhost', self.test_port, TEST_AE_TITLE)
    self.assertRaises(CouldNotCompleteDIMSEMessage,send_image,"NOT_SENDER_AE", address, DEFAULT_DATASET)

  def test_missing_sex(self):
    address = Address('localhost', self.test_port, TEST_AE_TITLE)
    ds = deepcopy(DEFAULT_DATASET)
    del ds.PatientSex
    response = send_image(SENDER_AE, address, ds)
    self.assertEqual(response.Status, 0xB006)

  def test_missing_PatientID(self):
    address = Address('localhost', self.test_port, TEST_AE_TITLE)
    ds = deepcopy(DEFAULT_DATASET)
    del ds.PatientID
    response = send_image(SENDER_AE, address, ds)
    self.assertEqual(response.Status, 0xB007)

  @bench
  def performance_send_concurrently(self):
    address = Address('localhost', self.test_port, TEST_AE_TITLE)
    images_1 = DicomTree(generate_numpy_datasets(50, PatientID = "1502799995"))
    images_2 = DicomTree(generate_numpy_datasets(50, PatientID = "0201919996"))

    images_1.map(personify(
      tags=[
        (0x00100010, "PN", "Odd Haugen Test"),
        (0x00100040, "CS", "M")
      ]
    ))

    images_2.map(personify(
      tags=[
        (0x00100010,"PN", "Ellen Louise Test"),
        (0x00100040,"CS", "M")
      ]
    ))


    thread_1 = send_images_thread(SENDER_AE, address, images_1, None, False)
    thread_2 = send_images_thread(SENDER_AE, address, images_2, None, False)

    ret_1 = thread_1.join()
    ret_2 = thread_2.join()

class FaultyNode(AbstractPipeline):
  ae_title = TEST_AE_TITLE
  input = {INPUT_KW : TestInput }
  require_calling_aet = [SENDER_AE]
  log_level: int = logging.CRITICAL
  disable_pynetdicom_logger: bool = True

  def process(self, InputData: InputContainer) -> PipelineOutput:
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

fs_path = Path('/tmp/pipeline_tests/fs')

class FileStorageTestNodeTestCase(TestCase):
  def setUp(self):
    fs_path.mkdir(parents=True, exist_ok=True)
    self.node = TestNode(start=False)
    self.test_port = randint(1025,65535)
    self.node.port = self.test_port
    self.node.open(blocking=False)

  @bench
  def performance_threaded_send_concurrently_fs(self):
    address = Address('localhost', self.test_port, TEST_AE_TITLE)
    images_1 = DicomTree(generate_numpy_datasets(50, PatientID = "1502799995"))
    images_2 = DicomTree(generate_numpy_datasets(50, PatientID = "0201919996"))

    images_1.map(personify(
      tags=[
        (0x00100010, "PN", "Odd Haugen Test"),
        (0x00100040, "CS", "M")
      ]
    ))

    images_2.map(personify(
      tags=[
        (0x00100010,"PN", "Ellen Louise Test"),
        (0x00100040,"CS", "M")
      ]
    ))

    thread_1 = send_images_thread(SENDER_AE, address, images_1, None, False)
    thread_2 = send_images_thread(SENDER_AE, address, images_2, None, False)

    ret_1 = thread_1.join()
    ret_2 = thread_2.join()

    self.assertEqual(ret_1, 0)
    self.assertEqual(ret_2, 0)

    self.assertEqual(self.node._AbstractPipeline__data_state.images,100) # type: ignore

  def test_threaded_send_concurrently_fs(self):
    address = Address('localhost', self.test_port, TEST_AE_TITLE)
    num_images = 2

    images_1 = DicomTree(generate_numpy_datasets(num_images, PatientID = "1502799995"))
    images_2 = DicomTree(generate_numpy_datasets(num_images, PatientID = "0201919996"))

    images_1.map(personify(
      tags=[
        (0x00100010, "PN", "Odd Haugen Test"),
        (0x00100040, "CS", "M")
      ]
    ))

    images_2.map(personify(
      tags=[
        (0x00100010,"PN", "Ellen Louise Test"),
        (0x00100040,"CS", "M")
      ]
    ))

    thread_1 = send_images_thread(SENDER_AE, address, images_1, None, False)
    thread_2 = send_images_thread(SENDER_AE, address, images_2, None, False)

    ret_1 = thread_1.join()
    ret_2 = thread_2.join()

    self.assertEqual(ret_1, 0)
    self.assertEqual(ret_2, 0)

    self.assertEqual(self.node._AbstractPipeline__data_state.images,2 * num_images) # type: ignore


class MaxFilterNode(AbstractPipeline):
  ae_title = TEST_AE_TITLE
  input = {INPUT_KW : TestInput }
  require_calling_aet = [SENDER_AE]
  log_level: int = logging.CRITICAL
  disable_pynetdicom_logger: bool = True

  def filter(self, dataset: Dataset) -> bool:
    return False

  def process(self, InputData: InputContainer) -> PipelineOutput:
    raise Exception


class MaxFilterTestCase(TestCase):
  def setUp(self):
    self.node = MaxFilterNode(start=False)
    self.test_port = randint(1025,65535)
    self.node.port = self.test_port
    self.node.open(blocking=False)

  def tearDown(self) -> None:
    self.node.close()

  def test_send_C_store_rejected(self):
    address = Address('localhost', self.test_port, TEST_AE_TITLE)
    response = send_image(SENDER_AE, address, DEFAULT_DATASET)
    self.assertEqual(response.Status, 0xB006)


class FaultyFilterNode(AbstractPipeline):
  ae_title = TEST_AE_TITLE
  input = {INPUT_KW : TestInput }
  require_calling_aet = [SENDER_AE]
  log_level: int = logging.CRITICAL
  disable_pynetdicom_logger: bool = True

  def filter(self, dataset: Dataset) -> bool:
    raise Exception

  def process(self, InputData: InputContainer) -> PipelineOutput:
    raise Exception


class FaultyFilterTestCase(TestCase):
  def setUp(self):
    self.node = FaultyFilterNode(start=False)
    self.test_port = randint(1025,65535)
    self.node.port = self.test_port
    self.node.open(blocking=False)

  def tearDown(self) -> None:
    self.node.close()

  def test_send_C_store_rejected(self):
    address = Address('localhost', self.test_port, TEST_AE_TITLE)
    response = send_image(SENDER_AE, address, DEFAULT_DATASET)
    self.assertEqual(response.Status, 0xA801)