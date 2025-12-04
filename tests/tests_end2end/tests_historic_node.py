"""This is a test case of a historic input"""

# Python3 standard library
from datetime import date
from random import randint
import logging
from time import sleep
from sys import stdout
from unittest import TestCase, skip

# Third party Packages
from pydicom import Dataset
from pydicom.uid import SecondaryCaptureImageStorage
from pynetdicom import build_context

# Dicomnode Packages
from dicomnode.dicom import gen_uid, make_meta
from dicomnode.dicom.dimse import Address, send_image, create_query_dataset,\
  QueryLevels
from dicomnode.server.nodes import AbstractPipeline
from dicomnode.server.pipeline_tree import InputContainer
from dicomnode.server.output import PipelineOutput, NoOutput
from dicomnode.server.input import HistoricAbstractInput

# Testing Packages
from tests.helpers import get_test_ae
from tests.helpers.storage_endpoint import TestStorageEndpoint, ENDPOINT_PORT
from tests.helpers.inputs import NeverValidatingInput
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
historic_dataset.SeriesDescription = "Historic"
make_meta(historic_dataset)


class TestCaseStorageEndpoint(TestStorageEndpoint):
  def handle_C_find(self, evt): #type: ignore
    yield (0xFF00, historic_dataset)

  def handle_C_move(self, evt): # type: ignore
    ip, port = self.move_target
    yield ip, port, { "contexts" : [build_context(historic_dataset.SOPClassUID)], "ae_title" : TEST_AE_TITLE}
    yield 1
    yield (0xFF00, historic_dataset)




class TestHistoricInput(HistoricAbstractInput):
  address = Address('localhost', ENDPOINT_PORT, "DUMMY")

  def check_query_dataset(self, current_study: Dataset) -> Dataset | None:
    return create_query_dataset(QueryLevels.PATIENT, PatientID=test_patient_id)

  def handle_found_dataset(self, found_dataset: Dataset) -> Dataset | None:
    return create_query_dataset(QueryLevels.PATIENT, PatientID=test_patient_id)


class HistoricPipeline(AbstractPipeline):
  ae_title = TEST_AE_TITLE
  input = {
    INPUT_KW : NeverValidatingInput,
    HISTORIC_KW : TestHistoricInput
  }
  require_calling_aet = []
  log_level: int = logging.DEBUG
  disable_pynetdicom_logger: bool = False
  pynetdicom_logger_level = logging.ERROR
  processing_directory = None
  log_output = None

  def process(self, input_data: InputContainer) -> PipelineOutput:
    return NoOutput()

class HistoricTestCase(DicomnodeTestCase):
  def setUp(self) -> None:
    self.node = HistoricPipeline()
    self.test_port = randint(1025,65535)

    self.node.port = self.test_port
    self.node.open(blocking=False)

    self.endpoint = TestCaseStorageEndpoint(move_endpoint=self.test_port)
    self.endpoint.open()

  def tearDown(self) -> None:
    self.node.close()
    self.endpoint.close()


  def test_end_2_end_historic_input(self):
    address = Address("localhost", self.test_port, TEST_AE_TITLE)

    dataset = Dataset()
    dataset.SOPInstanceUID = gen_uid()
    dataset.SOPClassUID = SecondaryCaptureImageStorage
    dataset.StudyDate = current_date
    dataset.PatientID = test_patient_id

    make_meta(dataset)


    with self.assertLogs(self.node.logger, logging.DEBUG) as cm:
      response = send_image(SENDER_AE_TITLE, address, dataset)
      sleep(3.25) # wait for all the threads to be done

    logs = "\n".join(cm.output)

    print(logs)
