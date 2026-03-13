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
import re
from typing import List, Dict, Any, Iterable, NoReturn, Optional, Tuple
import threading
from unittest import skip, TestCase

# Third Party packages #
import numpy
from pydicom import Dataset, DataElement
from pydicom.uid import RawDataStorage, SecondaryCaptureImageStorage

# Dicomnode Packages
from dicomnode.constants import DICOMNODE_LOGGER_NAME
from dicomnode.dicom import gen_uid, make_meta
from dicomnode.dicom.dicom_factory import DicomFactory
from dicomnode.dicom.series import DicomSeries
from dicomnode.dicom.blueprints.error_blueprint_english import ERROR_BLUEPRINT
from dicomnode.lib.io import Directory
from dicomnode.server.input import AbstractInput
from dicomnode.server.nodes import AbstractPipeline
from dicomnode.server.grinders import SeriesGrinder
from dicomnode.server.output import PipelineOutput, NoOutput
from dicomnode.server.pipeline_tree import InputContainer
from dicomnode.server.factories.association_events import CStoreEvent,\
  ReleasedEvent, AssociationTypes
from dicomnode.server.processor import AbstractProcessor

# Test Helpers #
from tests.helpers.dicomnode_test_case import DicomnodeTestCase
from tests.helpers import TESTING_TEMPORARY_DIRECTORY, generate_numpy_datasets,\
  clear_logger
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

DATASETS = DicomSeries([ds for ds in generate_numpy_datasets(
  11,
  Cols=11,
  Rows=12,
  PatientID=TEST_CPR
)])

DATASETS[0x0010_0040] = 'M'

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
    else:
      return False

    return True

  def add_image(self, dicom: Dataset) -> int:
    if 0x00110102 in dicom:
      raise Exception

    return super().add_image(dicom)


class RaisingProcessor(AbstractProcessor):
  def process(self, input_container: InputContainer) -> PipelineOutput:
    raise Exception("OOOOOH NOES")

class NoOpProcessor(AbstractProcessor):
  def process(self, input_container: InputContainer) -> PipelineOutput:
    return NoOutput()

class TestPipeLine(AbstractPipeline):
  data_directory = DICOM_STORAGE_PATH
  processing_directory = PROCESSING_DIRECTORY
  log_level = logging.DEBUG
  log_output = "test_file.log"

  Processor = RaisingProcessor

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

  def __init__(self) -> None:
    super().__init__()
    self.raise_error = False

class PipeLineTestCase(DicomnodeTestCase):
  def setUp(self) -> None:
    self.node = TestPipeLine()
    self.node.log_output = f"{self._testMethodName}.log"
    self.node._setup_logger()
    self.thread_id = threading.get_native_id()

  def tearDown(self) -> None:
    clear_logger(DICOMNODE_LOGGER_NAME)

    super().tearDown()







    # Note that you can't use the normal context handler to figure this out

  def test_exception_handler(self):
    class HandlerPipeline(AbstractPipeline):
      unhandled_error_blueprint = ERROR_BLUEPRINT
      default_response_port = 11112
      class Processor(AbstractProcessor):
        def process(self, input_container: InputContainer) -> PipelineOutput:
          return super().process(input_container)

    node = HandlerPipeline()
    with self.assertLogs(node.logger) as recorded_log:
      node.exception_handler_respond_with_dataset(
        Exception(),
        ReleasedEvent(
          1, None, "TEST", set()
        ),
        InputContainer({},{},None)
      )

    self.assertIn('ERROR:dicomnode:Unable to send error dataset to client due'
                  ' to missing IP address',recorded_log.output)

    with self.assertLogs(node.logger) as recorded_log:
      node.exception_handler_respond_with_dataset(
        Exception(),
        ReleasedEvent(
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
        ReleasedEvent(
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
        ReleasedEvent(
          1, '127.0.0.1', "TEST", set()
        ),
        InputContainer({},{
          'DATASETS' : [
            dataset
          ]
        },None)
      )

    endpoint.close()
