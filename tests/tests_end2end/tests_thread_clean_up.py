

# Python standard library
from pprint import pprint
from logging import DEBUG
from random import randint
from time import sleep

# Third party modules
from pydicom import Dataset
from pynetdicom.association import Association

# Dicomnode Modules
from dicomnode.lib.logging import get_logger
from dicomnode.constants import DICOMNODE_LOGGER_NAME
from dicomnode.dicom.dimse import send_images, Address, CouldNotCompleteDIMSEMessage
from dicomnode.server.input import AbstractInput
from dicomnode.server.pipeline_storage import ReactivePipelineStorage
from dicomnode.server.nodes import AbstractPipeline

# Test helper modules
from tests.helpers import DummyProcessor, generate_numpy_datasets, clear_logger
from tests.helpers.dicomnode_test_case import DicomnodeTestCase

class FailOnAddInput(AbstractInput):
  def validate(self) -> bool:
    return False

  def add_image(self, dicom: Dataset) -> int:
    logger = get_logger()
    raise Exception("Did you read the can?")

class FailOnValidate(AbstractInput):
  def validate(self) -> bool:
    raise Exception("I'm the problem")


AE = "VICTIM"
PATIENT_ID = "Patient_ID"

class VictimPipeline(AbstractPipeline):
  ae_title = AE

  input = {
    "asdf" : FailOnAddInput
  }

  Processor = DummyProcessor

class ThreadingCleanUpOnErrorTestCase(DicomnodeTestCase):
  def test_threading_fail_clean_up(self):
    """This test is for testing that when you send an image, and it fails
       unexpectedly, in validation / add an image, that either a other thread
       can clean up or something somewhere fixes this.

       So step 1 is to create a test case, that shows the problems
    """
    port = randint(1025, 45000)

    pipeline = VictimPipeline()
    with self.assertLogs(DICOMNODE_LOGGER_NAME):
      pipeline.port = port
      pipeline.input = { "asdf" : FailOnAddInput }

      pipeline.open(False)
      datasets = list(generate_numpy_datasets(3, Cols=10, Rows=10, PatientID=PATIENT_ID))

      try:
        send_images(
          "TESTCASE",
          Address('127.0.0.1', port, AE),
          datasets,
          logger= get_logger()
        )
      except CouldNotCompleteDIMSEMessage:
        pass

      pipeline.close()

    # Note that here we don't really care
    if isinstance(pipeline.data_state, ReactivePipelineStorage):
      for heartbeat in pipeline.data_state.thread_registration[PATIENT_ID]:
        self.assertFalse(heartbeat.is_active())

    clear_logger(DICOMNODE_LOGGER_NAME)
