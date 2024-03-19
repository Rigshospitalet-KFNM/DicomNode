"""These are test for a node. AVOID end 2 end test cases here, I set up a server
Most of these tests are just to show case the many error states, and that the
node logs correctly"""

__author__ = "Christoffer Vilstrup Jensen"

# Standard Python Library #
import logging
from pathlib import Path
from sys import stdout

from typing import List, Dict, Any, Iterable, NoReturn, Optional, Tuple
import threading
from unittest import skip, TestCase

# Third Party packages #
import numpy
from pydicom import Dataset
from pydicom.uid import RawDataStorage, SecondaryCaptureImageStorage

# Dicomnode Packages
from dicomnode.dicom import gen_uid, make_meta
from dicomnode.dicom.dicom_factory import DicomFactory
from dicomnode.dicom.series import DicomSeries
from dicomnode.server.input import AbstractInput
from dicomnode.server.nodes import AbstractPipeline
from dicomnode.server.output import PipelineOutput
from dicomnode.server.pipeline_tree import InputContainer
from dicomnode.server.factories.association_container import CStoreContainer,\
  ReleasedContainer, AssociationTypes

# Test Helpers #
from tests.helpers import TESTING_TEMPORARY_DIRECTORY

# Constants declarations #
TEST_AE_TITLE = "NODE_TITLE"
SENDER_AE_TITLE = "SENDER_TITLE"

DEFAULT_DATASET = Dataset()
DEFAULT_DATASET.SOPClassUID = RawDataStorage
DEFAULT_DATASET.PatientSex = 'M'
make_meta(DEFAULT_DATASET)

ENDPOINT_PORT = 50000
ENDPOINT_AE = "ENDPOINT_AT"

TEST_CPR = "1502799995"
INPUT_KW = "test_input"
HISTORIC_KW = "historic_input"

DEFAULT_DATASET.PatientID = TEST_CPR

DICOM_STORAGE_PATH = Path(f"{TESTING_TEMPORARY_DIRECTORY}/file_storage")
PROCESSING_DIRECTORY = Path(f"{TESTING_TEMPORARY_DIRECTORY}/working_directory")

class TestInput(AbstractInput):
  required_values = {
    0x00100040 : 'M'
  }

  def validate(self) -> bool:
    data = self.get_datasets()
    if len(data):
      pivot = data[0]

      print(pivot)
      if 0x00100021 in pivot:
        return False

    return True

  def add_image(self, dicom: Dataset) -> int:
    if 0x00110102 in dicom:
      raise Exception

    return super().add_image(dicom)


class TestPipeLine(AbstractPipeline):
  data_directory = DICOM_STORAGE_PATH
  processing_directory = PROCESSING_DIRECTORY
  log_level = logging.DEBUG
  log_output = stdout

  input = {
    INPUT_KW : TestInput
  }

  def filter(self, dataset) -> bool:
    if 0x00110101 in dataset:
      if dataset[0x00110101].value == 'ret_false':
        return False
      if dataset[0x00110101].value == 'ret_raise':
        raise Exception
    return True


class PipeLineTestCase(TestCase):
  def setUp(self) -> None:
    self.node = TestPipeLine()
    self.thread_id = threading.get_native_id()


  def tearDown(self) -> None:
    pass

  def test_consume_c_s_missing_Patient_ID(self):
    # Setup
    self.node._updated_patients[self.thread_id] = set()
    input_dataset = Dataset()

    container = CStoreContainer(
      association_id=self.thread_id,
      association_ip=None,
      association_ae="AE",
      dataset=input_dataset
    )

    with self.assertLogs('dicomnode', logging.DEBUG) as cm:
      self.assertEqual(0xB007,self.node.consume_c_store_container(container))

    self.assertEqual(cm.output[0], "INFO:dicomnode:Node rejected dataset: Received dataset doesn't have patient Identifier tag")

  def test_consume_c_store_missing_required_tag(self):
    self.node._updated_patients[self.thread_id] = set()
    input_dataset = Dataset()
    input_dataset.PatientID = TEST_CPR

    container = CStoreContainer(
      association_id=self.thread_id,
      association_ip=None,
      association_ae="AE",
      dataset=input_dataset
    )

    with self.assertLogs('dicomnode', logging.DEBUG) as cm:
      self.assertEqual(0xB006,self.node.consume_c_store_container(container))

    self.assertIn("INFO:dicomnode:Node rejected dataset: Received dataset is not accepted by any inputs", cm.output)

  def test_consume_c_store_fails_filter(self):
    self.node._updated_patients[self.thread_id] = set()
    input_dataset = Dataset()
    input_dataset.PatientID = TEST_CPR
    input_dataset.add_new(0x00110101,'LO', 'ret_false')

    container = CStoreContainer(
      association_id=self.thread_id,
      association_ip=None,
      association_ae="AE",
      dataset=input_dataset
    )

    with self.assertLogs('dicomnode', logging.DEBUG) as cm:
      self.assertEqual(0xB006,self.node.consume_c_store_container(container))

    self.assertIn("INFO:dicomnode:Node rejected dataset: Received Dataset did not pass filter", cm.output)

  def test_consume_c_store_exception_filter(self):
    self.node._updated_patients[self.thread_id] = set()
    input_dataset = Dataset()
    input_dataset.PatientID = TEST_CPR
    input_dataset.add_new(0x00110101,'LO', 'ret_raise')

    container = CStoreContainer(
      association_id=self.thread_id,
      association_ip=None,
      association_ae="AE",
      dataset=input_dataset
    )

    with self.assertLogs('dicomnode', logging.DEBUG) as cm:
      self.assertEqual(0xA801, self.node.consume_c_store_container(container))

    self.assertIn("CRITICAL:dicomnode:User defined function filter produced an exception", cm.output)

  def test_consume_c_store_add_image_raises(self):
    self.node._updated_patients[self.thread_id] = set()
    input_dataset = Dataset()
    input_dataset.PatientID = TEST_CPR
    input_dataset.add_new(0x00110102,'LO', 'ret_raise')

    container = CStoreContainer(
      association_id=self.thread_id,
      association_ip=None,
      association_ae="AE",
      dataset=input_dataset
    )

    with self.assertLogs('dicomnode', logging.DEBUG) as cm:
      self.assertEqual(0xA801, self.node.consume_c_store_container(container))

    self.assertIn("CRITICAL:dicomnode:Adding Image to input produced an exception", cm.output)

  def test_consume_c_store_success(self):
    self.node._updated_patients[self.thread_id] = set()
    input_dataset = Dataset()
    input_dataset.PatientID = TEST_CPR
    input_dataset.SOPInstanceUID = gen_uid()
    input_dataset.SOPClassUID = SecondaryCaptureImageStorage
    input_dataset.PatientSex = 'M'
    make_meta(input_dataset)

    container = CStoreContainer(
      association_id=self.thread_id,
      association_ip=None,
      association_ae="AE",
      dataset=input_dataset
    )

    with self.assertNoLogs('dicomnode', logging.DEBUG):
      out = self.node.consume_c_store_container(container)

    self.assertEqual(out, 0)
    self.assertIn(TEST_CPR, self.node._patient_locks)
    self.assertIn(TEST_CPR, self.node._updated_patients[self.thread_id])
    self.assertEqual(self.node.data_state.images, 1)

  def test_consume_release_container(self):
    self.node._updated_patients[self.thread_id] = set([TEST_CPR])
    self.node._patient_locks[TEST_CPR] = (set([self.thread_id]), threading.Lock())

    factory = DicomFactory()

    self.patient_name = "test^patient"
    info_uint16 = numpy.iinfo(numpy.uint16)
    def gen_dataset(i: int) -> Dataset:
      ds = Dataset()
      factory.store_image_in_dataset(ds, numpy.random.randint(0,info_uint16.max, (11,12), numpy.uint16))
      ds.InstanceNumber = i + 1
      ds.PatientName = self.patient_name
      ds.PatientID = TEST_CPR
      ds.PatientSex = 'M'
      return ds

    parent_series = DicomSeries([
      gen_dataset(i) for i in range(13)
    ])

    self.node.data_state.add_images(parent_series.datasets)

    container = ReleasedContainer(
      association_id=self.thread_id,
      association_ip=None,
      association_ae="AE",
      association_types=[AssociationTypes.STORE_ASSOCIATION]
    )


    with self.assertLogs('dicomnode', logging.DEBUG) as cm:
      self.node._consume_association_release_store_association(container)
    for string in cm.output:
      #print(string)
      pass

  def test_consume_release_container_not_enough_data(self):
    self.node._updated_patients[self.thread_id] = set([TEST_CPR])
    self.node._patient_locks[TEST_CPR] = (set([self.thread_id]), threading.Lock())

    factory = DicomFactory()

    self.patient_name = "test^patient"
    info_uint16 = numpy.iinfo(numpy.uint16)
    def gen_dataset(i: int) -> Dataset:
      ds = Dataset()
      factory.store_image_in_dataset(ds, numpy.random.randint(0,info_uint16.max, (11,12), numpy.uint16))
      ds.InstanceNumber = i + 1
      ds.PatientName = self.patient_name
      ds.PatientID = TEST_CPR
      ds.PatientSex = 'M'
      ds.StudyDescription = "WHY U NO WORK"
      return ds

    parent_series = DicomSeries([
      gen_dataset(i) for i in range(13)
    ])

    self.node.data_state.add_images(parent_series.datasets)

    container = ReleasedContainer(
      association_id=self.thread_id,
      association_ip=None,
      association_ae="AE",
      association_types=[AssociationTypes.STORE_ASSOCIATION]
    )


    with self.assertLogs('dicomnode', logging.DEBUG) as cm:
      self.node._consume_association_release_store_association(container)
    for string in cm.output:
      #print(string)
      pass

  def test_consume_release_container_another_thread_leaving(self):
    self.node._updated_patients[self.thread_id] = set([TEST_CPR])
    self.node._patient_locks[TEST_CPR] = (set([self.thread_id,
                                               self.thread_id + 1]),
                                          threading.Lock())

    factory = DicomFactory()

    self.patient_name = "test^patient"
    info_uint16 = numpy.iinfo(numpy.uint16)
    def gen_dataset(i: int) -> Dataset:
      ds = Dataset()
      factory.store_image_in_dataset(ds, numpy.random.randint(0,info_uint16.max, (11,12), numpy.uint16))
      ds.InstanceNumber = i + 1
      ds.PatientName = self.patient_name
      ds.PatientID = TEST_CPR
      ds.PatientSex = 'M'
      return ds

    parent_series = DicomSeries([
      gen_dataset(i) for i in range(13)
    ])

    self.node.data_state.add_images(parent_series.datasets)

    container = ReleasedContainer(
      association_id=self.thread_id,
      association_ip=None,
      association_ae="AE",
      association_types=[AssociationTypes.STORE_ASSOCIATION]
    )


    with self.assertLogs('dicomnode', logging.DEBUG) as cm:
      self.node._consume_association_release_store_association(container)
    for string in cm.output:
      #print(string)
      pass

  def test_dispatch(self):
    class DumbOutput(PipelineOutput):
      def __init__(self, val) -> None:
        self.val = val

      def send(self):
        return self.val

    self.assertTrue(self.node._dispatch(
      DumbOutput(True)
    ))
    self.assertFalse(self.node._dispatch(
      DumbOutput(False)
    ))

  def test_dispatch_raises(self):
    class RasingOutput(PipelineOutput):
      def __init__(self ) -> None:
        pass

      def send(self):
        raise Exception

    with self.assertLogs('dicomnode') as cm:
      self.assertFalse(self.node._dispatch(RasingOutput()))

    self.assertIn("CRITICAL:dicomnode:Exception in user Output Send Function",
                  cm.output)
