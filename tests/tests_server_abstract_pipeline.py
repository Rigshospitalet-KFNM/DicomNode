"""These are test for a node. AVOID end 2 end test cases here, I set up a server
Most of these tests are just to show case the many error states, and that the
node logs correctly

Another critical difference is that they do not setup a server. They just call
different handler functions

For the tests 
"""

__author__ = "Christoffer Vilstrup Jensen"

# Standard Python Library #
import logging
from logging import DEBUG
from pathlib import Path
from sys import stdout
from shutil import rmtree
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
from dicomnode.dicom.blueprints.error_blueprint_english import ERROR_BLUEPRINT
from dicomnode.server.input import AbstractInput
from dicomnode.server.nodes import AbstractPipeline
from dicomnode.server.grinders import SeriesGrinder
from dicomnode.server.output import PipelineOutput, NoOutput
from dicomnode.server.pipeline_tree import InputContainer
from dicomnode.server.factories.association_container import CStoreContainer,\
  ReleasedContainer, AssociationTypes

# Test Helpers #
from tests.helpers import TESTING_TEMPORARY_DIRECTORY
from tests.helpers.storage_endpoint import TestStorageEndpoint

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

  image_grinder = SeriesGrinder()

  def validate(self) -> bool:
    data = self.get_datasets()
    if len(data):
      pivot = data[0]

      if 0x00110103 in pivot:
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

  def process(self, input_container: InputContainer):
    return NoOutput()


class PipeLineTestCase(TestCase):
  def setUp(self) -> None:
    self.node = TestPipeLine()
    self.thread_id = threading.get_native_id()

  def tearDown(self) -> None:
    storage = self.node.get_storage_directory(TEST_CPR)
    if storage.exists():
      rmtree(storage)

  def test_consume_c_store_missing_Patient_ID(self):
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

    self.assertEqual(cm.output[0], "INFO:dicomnode:Node rejected dataset: Received dataset doesn't have patient Identifier tag: 0x100020")

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

    log_process = f"INFO:dicomnode:Processing {TEST_CPR}"
    log_dispatch = f"INFO:dicomnode:Dispatched {TEST_CPR} Successful"
    self.assertIn(log_process, cm.output)
    self.assertIn(log_dispatch, cm.output)

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
      ds.add_new(0x00110103, 'LO', "This works now?")
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

    log = f"INFO:dicomnode:Insufficient data for patient {TEST_CPR}"
    self.assertIn(log, cm.output)


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
      factory.store_image_in_dataset(ds, numpy.random.randint(0,
                                                              info_uint16.max,
                                                              (11,12),
                                                              numpy.uint16)
                                    )
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

    log = f"DEBUG:dicomnode:PatientID to be updated in: {{{self.thread_id}: {{'1502799995'}}}}"
    self.assertIn(log, cm.output)


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

  def test_missing_output_from_process(self):
    class DumbPipeline(AbstractPipeline):
      def process(self, *args):
        return None
    node = DumbPipeline()
    with self.assertLogs(node.logger, DEBUG) as recorded_logs:
      node._pipeline_processing(
        'Test_patient',
        ReleasedContainer(0, '127.0.0.1', "TEST", ()),
        InputContainer({}, {}, None)
      )
    self.assertIn('WARNING:dicomnode:You forgot to return a PipelineOutput '
                  'object in the process function. If output is handled by '
                  'process, return a NoOutput Object', recorded_logs.output)

  def test_double_failed_log(self):
    class FalseOutput(PipelineOutput):
      def __init__(self ) -> None:
        pass

      def send(self):
        return False

    class DumbPipeline(AbstractPipeline):
      def process(self, *args):
        return FalseOutput()

    node = DumbPipeline()
    with self.assertLogs(node.logger, DEBUG) as recorded_logs:
      node._pipeline_processing(
        'Test_patient',
        ReleasedContainer(0, '127.0.0.1', "TEST", ()),
        InputContainer({}, {}, None)
      )
    self.assertIn('ERROR:dicomnode:Unable to dispatch output for Test_patient', recorded_logs.output)


  def test_exception_handler(self):
    class HandlerPipeline(AbstractPipeline):
      unhandled_error_blueprint = ERROR_BLUEPRINT
      default_response_port = 11112

    node = HandlerPipeline()
    with self.assertLogs(node.logger) as recorded_log:
      node.exception_handler_respond_with_dataset(
        Exception(),
        ReleasedContainer(
          1, None, "TEST", set()
        ),
        InputContainer({},{},None)
      )

    self.assertIn('ERROR:dicomnode:Unable to send error dataset to client due'
                  ' to missing IP address',recorded_log.output)

    with self.assertLogs(node.logger) as recorded_log:
      node.exception_handler_respond_with_dataset(
        Exception(),
        ReleasedContainer(
          1, '127.0.0.1', "TEST", set()
        ),
        InputContainer({},{},None)
      )

    self.assertIn('ERROR:dicomnode:Unable to extract a dataset from the input '
                  'container',recorded_log.output)

    dataset = Dataset()
    dataset.SOPClassUID = SecondaryCaptureImageStorage
    dataset.AccessionNumber = "TEST_NUMBER"
    dataset.PatientName = "PATIENT NAME"
    dataset.PatientID = "123456970"
    dataset.StudyInstanceUID = gen_uid()

    with self.assertLogs(node.logger) as recorded_log:
      node.exception_handler_respond_with_dataset(
        Exception(),
        ReleasedContainer(
          1, '127.0.0.1', "TEST", set()
        ),
        InputContainer({},{
          'DATASETS' : [
            dataset
          ]
        },None)
      )
    self.assertIn(
      'ERROR:dicomnode:Unable to send error message to the client at'
      ' 127.0.0.1:11112 - TEST', recorded_log.output
    )

    endpoint = TestStorageEndpoint(11112, "ENDPOINT")
    endpoint.open()

    with self.assertLogs(node.logger) as recorded_log:
      node.exception_handler_respond_with_dataset(
        Exception(),
        ReleasedContainer(
          1, '127.0.0.1', "TEST", set()
        ),
        InputContainer({},{
          'DATASETS' : [
            dataset
          ]
        },None)
      )

    endpoint.close()
