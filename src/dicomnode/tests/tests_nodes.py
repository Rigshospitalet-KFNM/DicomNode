from unittest import TestCase
from dicomnode.lib.dimse import send_images
from random import randint

from dicomnode.server.input import AbstractInput
from dicomnode.server.nodes import AbstractPipeline

from pydicom import Dataset
from pydicom.uid import generate_uid, RawDataStorage, ImplicitVRLittleEndian

from typing import List, Dict, Any

TEST_AE_TITLE = "TEST_AE"

class TestInput(AbstractInput):
  required_tags: List[int] = [0x00080018]

  def validate(self):
    return True


class NodeTestImplementation(AbstractPipeline):
  ae_title = TEST_AE_TITLE
  input = {'test_input' : TestInput }

  def process(self, InputData: Dict[str, Any]):
    return


class Server_Nodes_TestCase(TestCase):
  def setUp(self):
    self.node = NodeTestImplementation(start=False)
    self.testPort = randint(1025,65535)
    self.node.port = self.testPort
    self.node.open(blocking=False)

  def tearDown(self) -> None:
    self.node.close()

  def test_send_C_store_missing_patient_identifier(self):
    sender_AE = "SENDER_AE"

    ds = Dataset()
    ds.SOPInstanceUID = generate_uid()
    ds.SOPClassUID = RawDataStorage
    ds.TransferSyntaxUID = ImplicitVRLittleEndian
    ds.fix_meta_info()
    ds.file_meta.TransferSyntaxUID = ImplicitVRLittleEndian

    response = send_images('localhost', self.testPort, TEST_AE_TITLE, sender_AE, ds)
    self.assertEqual(response.Status, 0xB007)

  def test_send_C_store_success(self):
    sender_AE = "SENDER_AE"

    ds = Dataset()
    ds.SOPInstanceUID = generate_uid()
    ds.SOPClassUID = RawDataStorage
    ds.TransferSyntaxUID = ImplicitVRLittleEndian
    ds.PatientID = "1502799995"
    ds.fix_meta_info()
    ds.file_meta.TransferSyntaxUID = ImplicitVRLittleEndian

    response = send_images('localhost', self.testPort, TEST_AE_TITLE, sender_AE, ds)
    self.assertEqual(response.Status, 0x0000)

    PT = self.node._AbstractPipeline__data_state

