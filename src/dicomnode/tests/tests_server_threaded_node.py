import logging
from random import randint
from pathlib import Path
from pydicom import Dataset
from time import sleep, perf_counter
from unittest import TestCase

from dicomnode.lib.dimse import Address, send_images_thread
from dicomnode.lib.exceptions import CouldNotCompleteDIMSEMessage
from dicomnode.lib.imageTree import DicomTree

from dicomnode.tests.helpers import generate_numpy_datasets, personify, bench

from dicomnode.server.pipelineTree import InputContainer
from dicomnode.server.input import AbstractInput
from dicomnode.server.output import NoOutput, PipelineOutput
from dicomnode.server.threaded_node import AbstractThreadedPipeline
from pydicom.uid import RawDataStorage, ImplicitVRLittleEndian

from typing import List, Dict, Any, Iterable

TEST_AE_TITLE = "TEST_AE"
SENDER_AE = "SENDER_AE"
INPUT_KW = "test_input"

class TestInput(AbstractInput):
  required_tags: List[int] = [0x00080018]

  def validate(self):
    return True

class TestNode(AbstractThreadedPipeline):
  ae_title = TEST_AE_TITLE
  input = {INPUT_KW : TestInput }
  require_calling_aet = [SENDER_AE]
  log_level: int = logging.CRITICAL
  disable_pynetdicom_logger: bool = True

  def process(self, InputData: InputContainer) -> PipelineOutput:
    self.logger.info("process is called")
    return NoOutput()


class TestNodeTestCase(TestCase):
  def setUp(self):
    self.node = TestNode(start=False)
    self.test_port = randint(1025,65535)
    self.node.port = self.test_port
    self.node.open(blocking=False)

  def tearDown(self) -> None:
    self.node._join_threads()
    while self.node.ae.active_associations != []:
      sleep(0.005)

    self.node.close()

  @bench
  def performance_threaded_send_concurrently(self):
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

  def test_threaded_send_concurrently(self):
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


fs_path = Path("/tmp/pipeline_tests/fs_threaded")

class FileStorageTestNode(AbstractThreadedPipeline):
  ae_title = TEST_AE_TITLE
  input = {INPUT_KW : TestInput }
  require_calling_aet = [SENDER_AE]
  log_level: int = logging.CRITICAL
  disable_pynetdicom_logger: bool = True
  root_data_directory = fs_path

  def process(self, InputData: InputContainer) -> PipelineOutput:
    self.logger.info("process is called")
    return NoOutput()

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
        (0x00100010, "PN", "Odd Name Test"),
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