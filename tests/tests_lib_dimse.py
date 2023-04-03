""""""

__author__ = "Christoffer Vilstrup Jensen"

# Python Standard Library
import logging
from pprint import pprint, pformat
from random import randint
from unittest import skip, TestCase

#Third party packages
from pydicom import Dataset
from pydicom.uid import SecondaryCaptureImageStorage
from pynetdicom import debug_logger

# Dicomnode packages
from dicomnode.lib.exceptions import InvalidQueryDataset, CouldNotCompleteDIMSEMessage
from dicomnode.lib.logging import get_logger, DEBUG
from dicomnode.lib.dicom import make_meta
from dicomnode.lib.dimse import send_move, Address, QueryLevels

# Dicomnode tests helpers
from tests.helpers import get_test_ae

logger = get_logger()

class DIMSETestCases(TestCase):
  TEST_CASE_AE = "TEST_CASE"

  def test_send_move(self):
    self.endpoint_port = randint(1025,65535)
    self.endpoint = get_test_ae(self.endpoint_port, self.endpoint_port, logger)
    self.address = Address('localhost', self.endpoint_port, "PYNETDICOM")

    self.dataset = Dataset()
    self.dataset.is_little_endian = True
    self.dataset.is_implicit_VR = True
    self.dataset.PatientID = "1235971155"
    self.dataset.SOPClassUID = SecondaryCaptureImageStorage
    self.dataset.add_new(0x00080052, 'CS', 'PATIENT')
    make_meta(self.dataset)


    with self.assertLogs(logger, DEBUG) as logs_records:
      send_move(self.TEST_CASE_AE, self.address, self.dataset)

    self.assertIn("DEBUG:dicomnode:Sending C move", logs_records.output)
    self.assertIn("INFO:dicomnode:Received C Move", logs_records.output)
    self.assertIn("INFO:dicomnode:Received C Store", logs_records.output)
    self.assertIn("INFO:dicomnode:Finished handling C Move", logs_records.output)

    self.endpoint.shutdown()

  def test_send_move_invalid_quires(self):
    address = Address('localhost', 4321, "PYNETDICOM") # Connection will never be made nor is it needed

    dataset = Dataset()

    self.assertRaises(InvalidQueryDataset, send_move, address, "Dummy", dataset, QueryLevels.PATIENT)
    self.assertRaises(InvalidQueryDataset, send_move, address, "Dummy", dataset, QueryLevels.STUDY)
    self.assertRaises(InvalidQueryDataset, send_move, address, "Dummy", dataset, QueryLevels.SERIES)

  def test_send_move_no_connection(self):
    address = Address('localhost', 4321, "PYNETDICOM") # Connection will fails as there's no end point

    dataset = Dataset()
    dataset.PatientID = "FooBar"

    logging.getLogger("pynetdicom").setLevel(logging.CRITICAL + 1)

    with self.assertLogs(logger, DEBUG) as log_records:
      self.assertRaises(CouldNotCompleteDIMSEMessage,send_move,"Dummy", address, dataset)


