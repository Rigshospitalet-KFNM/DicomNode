import logging
from random import randint
from pydicom import Dataset
from pydicom.uid import SecondaryCaptureImageStorage
from unittest import TestCase

from pynetdicom import debug_logger

from dicomnode.lib.dicom import make_meta
from dicomnode.lib.dimse import send_move, Address
from dicomnode.tests.helpers import get_test_ae

class DIMSETestCases(TestCase):
  TEST_CASE_AE = "TEST_CASE"

  def setUp(self) -> None:
    self.endpoint_port = randint(1025,65535)
    self.endpoint = get_test_ae(self.endpoint_port, self.endpoint_port, logging.getLogger("dicomnode"))
    self.address = Address('localhost', self.endpoint_port, "DUMMY")

    self.dataset = Dataset()
    self.dataset.is_little_endian = True
    self.dataset.is_implicit_VR = True
    self.dataset.PatientID = "1235971155"
    self.dataset.SOPClassUID = SecondaryCaptureImageStorage
    self.dataset.add_new(0x00080052, 'CS', 'PATIENT')
    make_meta(self.dataset)
    #debug_logger()

  def teardown(self) -> None:
    self.endpoint.shutdown()

  def test_send_move(self):
    with self.assertLogs("dicomnode", logging.DEBUG) as cm:
      send_move(self.TEST_CASE_AE, self.address, self.dataset)

