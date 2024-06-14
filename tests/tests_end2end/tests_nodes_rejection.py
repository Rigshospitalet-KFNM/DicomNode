"""This end to end test, check that a dicomnode reject unknown AE titles"""

# Python standard library
from logging import DEBUG
from random import randint
from time import sleep
from unittest import TestCase
# Third party modules
from pydicom.uid import (
  ExplicitVRLittleEndian,
  ImplicitVRLittleEndian,
  DeflatedExplicitVRLittleEndian,
  ExplicitVRBigEndian,
)
from pynetdicom import AE
from pynetdicom.sop_class import Verification

# Dicomnode Modules
from dicomnode.server.input import AbstractInput
from dicomnode.server.nodes import AbstractPipeline

# Test packages
from tests.helpers.inputs import NeverValidatingInput

#region Setup
class RejectionAETitle(AbstractPipeline):
  input = {"BAH" : NeverValidatingInput}
  ae_title = "REJECT"
  log_output=None
  require_called_aet = True
  require_calling_aet = ["AETITLE1"]

transfer_syntax = [
  ExplicitVRLittleEndian,
  ImplicitVRLittleEndian,
  DeflatedExplicitVRLittleEndian,
  ExplicitVRBigEndian,
]


#region TestCase
class RejectionTestCase(TestCase):
  def setUp(self):
    self.node = RejectionAETitle()
    self.port = randint(1025,65535)
    self.node.port = self.port


  def tearDown(self) -> None:
    while self.node.dicom_application_entry.active_associations != []:
      sleep(0.005) #pragma: no cover
    self.node.close()

  def test_rejection(self):
    with self.assertLogs(self.node.logger):
      self.node.open(blocking=False)
    ae = AE(ae_title="NOTKNOWN")
    ae.add_requested_context(Verification, transfer_syntax)
    with self.assertLogs('dicomnode', DEBUG) as recorded_logs:
      assoc = ae.associate('127.0.0.1', self.port, ae_title="NOT TARGET")
      self.assertFalse(assoc.is_established)
    self.assertIn(f'DEBUG:dicomnode:Connection NOTKNOWN rejected a connection', recorded_logs.output)

    with self.assertLogs('dicomnode', DEBUG) as recorded_logs:
      assoc = ae.associate('127.0.0.1', self.port, ae_title="REJECT")
      self.assertFalse(assoc.is_established)
    self.assertIn(f'DEBUG:dicomnode:Connection NOTKNOWN rejected a connection', recorded_logs.output)

    ae.ae_title = "AETITLE1"
    with self.assertLogs('dicomnode', DEBUG) as recorded_logs:
      assoc = ae.associate('127.0.0.1', self.port, ae_title="NOT TARGET")
      self.assertFalse(assoc.is_established)
    self.assertIn(f'DEBUG:dicomnode:Connection AETITLE1 rejected a connection', recorded_logs.output)

    with self.assertLogs(self.node.logger, DEBUG) as recorded_logs:
      assoc = ae.associate('127.0.0.1', self.port, ae_title="REJECT")
      self.assertTrue(assoc.is_established)
      responds = assoc.send_c_echo()
      self.assertEqual(responds.Status, 0x0000)
      assoc.release()
    self.assertIn(f'DEBUG:dicomnode:Connection AETITLE1 send an echo', recorded_logs.output)




