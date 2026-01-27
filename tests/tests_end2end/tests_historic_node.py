"""This is a test case of a historic input"""

# Python3 standard library
from datetime import date
from random import randint
import logging
from typing import Sequence

from time import sleep
from sys import stdout


# Third party Packages
from pydicom import Dataset
from pydicom.uid import SecondaryCaptureImageStorage
from pynetdicom import build_context

# Dicomnode Packages
from dicomnode.constants import DICOMNODE_LOGGER_NAME, DICOMNODE_PROCESS_LOGGER
from dicomnode.dicom import gen_uid, make_meta
from dicomnode.dicom.dimse import Address, send_image, create_query_dataset,\
  QueryLevels
from dicomnode.server.nodes import AbstractPipeline
from dicomnode.server.pipeline_tree import InputContainer
from dicomnode.server.output import PipelineOutput, NoOutput
from dicomnode.server.input import HistoricAbstractInput, AbstractInput
from dicomnode.server.processor import AbstractProcessor

# Testing Packages
from tests.helpers import get_test_ae, clear_logger
from tests.helpers.storage_endpoint import ENDPOINT_AE_TITLE, MOVE_ENDPOINT, TestStorageEndpoint, ENDPOINT_PORT
from tests.helpers.dicomnode_test_case import DicomnodeTestCase

# Constants
TEST_AE_TITLE = "TEST_AE"
SENDER_AE_TITLE = "SENDER_AE"
HISTORIC_AE_TITLE = "HISTORIC"

INPUT_KW = "input"
HISTORIC_KW = "historic"

historic_date = date(2000, 1, 1)
current_date = date(2020, 1, 1)

historic_uid = gen_uid()
test_patient_id = "Test ID"

historic_dataset = Dataset()
historic_dataset.StudyDate = historic_date
historic_dataset.SOPInstanceUID = historic_uid
historic_dataset.SOPClassUID = SecondaryCaptureImageStorage
historic_dataset.PatientID = test_patient_id
historic_dataset.PatientSex = "M"
historic_dataset.SeriesDescription = "Historic"
make_meta(historic_dataset)


class TestCaseStorageEndpoint(TestStorageEndpoint):
  def __init__(self, datasets: Sequence[Dataset], **kwargs) -> None:
    self.datasets = datasets
    super().__init__(**kwargs)

  def handle_C_find(self, evt): #type: ignore
    for dataset in self.datasets:
      yield (0xFF00, dataset)

  def handle_C_move(self, evt): # type: ignore
    ip, port = self.move_target
    yield ip, port, { "contexts" : [build_context(historic_dataset.SOPClassUID)], "ae_title" : TEST_AE_TITLE}
    yield len(self.datasets)
    for dataset in self.datasets:
      yield (0xFF00, dataset)


class PresentInput(AbstractInput):
  enforce_single_study_date = True

  def validate(self) -> bool:
    return True

class TestHistoricInput(HistoricAbstractInput):
  address = Address('localhost', ENDPOINT_PORT, "DUMMY")

  def check_query_dataset(self, current_study: Dataset) -> Dataset | None:
    return create_query_dataset(QueryLevels.PATIENT, PatientID=test_patient_id)

  def handle_found_dataset(self, found_dataset: Dataset) -> Dataset | None:
    return create_query_dataset(QueryLevels.PATIENT, PatientID=test_patient_id)


class HistoricRunner(AbstractProcessor):
  def process(self, input_container: InputContainer) -> PipelineOutput:
    historic = input_container[HISTORIC_KW]

    self.logger.info(f"HISTORIC PROCESSING WITH DATA: {historic}")

    return NoOutput()

class HistoricPipeline(AbstractPipeline):
  ae_title = TEST_AE_TITLE
  input = {
    INPUT_KW : PresentInput,
    HISTORIC_KW : TestHistoricInput
  }
  require_calling_aet = []
  log_level: int = logging.DEBUG
  disable_pynetdicom_logger: bool = False
  pynetdicom_logger_level = logging.ERROR
  processing_directory = None
  log_output = None
  Processor = HistoricRunner

  def process(self, input_data: InputContainer) -> PipelineOutput:
    historic = input_data[HISTORIC_KW]

    self.logger.info(f"HISTORIC PROCESSING WITH DATA: {historic}")

    return NoOutput()

class HistoricTestCase(DicomnodeTestCase):
  def tearDown(self) -> None:

    clear_logger(DICOMNODE_LOGGER_NAME)
    clear_logger(DICOMNODE_PROCESS_LOGGER)

    super().tearDown()


  def test_end_2_end_historic_input(self):
    with self.assertLogs(DICOMNODE_LOGGER_NAME):
      node = HistoricPipeline()
      test_port = randint(40000,49999)

      node.port = test_port
      node.open(blocking=False)
      with self.assertLogs(DICOMNODE_PROCESS_LOGGER):
        endpoint = TestCaseStorageEndpoint([historic_dataset], move_endpoint=test_port)
        endpoint.open()

        address = Address("localhost", test_port, TEST_AE_TITLE)
        dataset = Dataset()
        dataset.SOPInstanceUID = gen_uid()
        dataset.SOPClassUID = SecondaryCaptureImageStorage
        dataset.StudyDate = current_date
        dataset.PatientID = test_patient_id
        dataset.PatientSex = "M"

        make_meta(dataset)

        response = send_image(SENDER_AE_TITLE, address, dataset)
        sleep(0.005)
        while node.processes:
          sleep(0.005)

        node.close()
        endpoint.close()

  def test_end_2_end_historic_input_empty(self):
    with self.assertLogs(DICOMNODE_LOGGER_NAME):
      node = HistoricPipeline()
      test_port = randint(40000,49999)

      node.port = test_port
      node.open(blocking=False)
      with self.assertNonCapturingLogs(DICOMNODE_PROCESS_LOGGER):
        endpoint = TestCaseStorageEndpoint([], move_endpoint=test_port)
        endpoint.open()

        address = Address("localhost", test_port, TEST_AE_TITLE)

        dataset = Dataset()
        dataset.SOPInstanceUID = gen_uid()
        dataset.SOPClassUID = SecondaryCaptureImageStorage
        dataset.StudyDate = current_date
        dataset.PatientID = test_patient_id
        dataset.PatientSex = "M"

        make_meta(dataset)
        response = send_image(SENDER_AE_TITLE, address, dataset)
        sleep(1.25) # wait for all the threads to be done


        self.assertEqual(node.data_state.images, 0)
        node.close()
        endpoint.close()