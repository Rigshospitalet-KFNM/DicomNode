"""This is a test case of a historic input"""

# Python3 standard library
import logging
from pathlib import Path
from random import randint
from unittest import skip

# Third party Packages
from pydicom import Dataset
from pydicom.uid import SecondaryCaptureImageStorage

# Dicomnode Packages
from dicomnode.constants import DICOMNODE_LOGGER_NAME, DICOMNODE_PROCESS_LOGGER
from dicomnode.dicom import make_meta, gen_uid
from dicomnode.dicom.dimse import Address, send_images
from dicomnode.server.nodes import AbstractPipeline
from dicomnode.server.input import AbstractInput
from dicomnode.server.output import NoOutput

# Testing Packages
from helpers import process_thread_check_leak
from helpers.dicomnode_test_case import DicomnodeTestCase
from helpers import clear_logger

class LogFileIsWritten(DicomnodeTestCase):
  def tearDown(self) -> None:
    clear_logger(DICOMNODE_LOGGER_NAME)
    clear_logger(DICOMNODE_PROCESS_LOGGER)

    return super().tearDown()


  #@skip("Something doesn't work here")
  @process_thread_check_leak
  def test_end2end_log_file_written(self):

    test_port = randint(1024, 45000)

    class DumbInput(AbstractInput):
      def validate(self) -> bool:
        return True

    class LoggingPipeline(AbstractPipeline):
      port = test_port
      ae_title = "TEST"

      input = {
        "input" : DumbInput
      }

      log_output = "log.log"

      def process(self, input_data):
        self.logger.setLevel(10)
        self.logger.info("Can we find the file?")
        return NoOutput()

    instance = LoggingPipeline()
    with self.assertLogs(DICOMNODE_LOGGER_NAME, logging.DEBUG):
      instance.open(blocking=False)
      with self.assertNonCapturingLogs(DICOMNODE_PROCESS_LOGGER, logging.DEBUG):
        dataset = Dataset()
        dataset.SOPInstanceUID = gen_uid()
        dataset.SOPClassUID = SecondaryCaptureImageStorage
        dataset.PatientID = "Patient^ID"
        dataset.InstanceNumber = 1
        make_meta(dataset)

        send_images("SENDER", Address('127.0.0.1', test_port, "TEST"), [dataset])

        instance.close()

        self.assertTrue(Path("log.log").exists())
        with open('log.log', 'r') as log_file:
          lines = log_file.readlines()

        self.assertGreater(len(lines),0)
