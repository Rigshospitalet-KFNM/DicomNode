from unittest import TestCase
from dicomnode.lib.dimse import send_c_store
from random import randint

from dicomnode.server.nodes import AbstractPipeline

from pydicom import Dataset
from pydicom.uid import generate_uid, RawDataStorage, ImplicitVRLittleEndian

from time import sleep

TEST_AE_TITLE = "TESTAE"
TEST_PORT = 11112


class NodeTestImplementation(AbstractPipeline):
  ae_title = TEST_AE_TITLE
  log_path = ""
  port = TEST_PORT


class Server_Nodes_TestCase(TestCase):
  def setUp(self):
    self.node = NodeTestImplementation(start=False)
    self.testPort = randint(32768,65535)
    self.node.port = self.testPort
    self.node.open(blocking=False)

  def tearDown(self) -> None:
    self.node.close()

  def test_startServer(self):
    sleep(0.25)

  def test_send_C_store_to_self(self):

    sender_AE = "SENDER_AE"

    ds = Dataset()
    ds.SOPInstanceUID = generate_uid()
    ds.SOPClassUID = RawDataStorage
    ds.TransferSyntaxUID = ImplicitVRLittleEndian
    ds.fix_meta_info()
    ds.file_meta.TransferSyntaxUID = ImplicitVRLittleEndian

    send_c_store('localhost', self.testPort, TEST_AE_TITLE, sender_AE, ds)
