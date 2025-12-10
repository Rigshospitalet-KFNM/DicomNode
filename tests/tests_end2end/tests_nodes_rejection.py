"""This end to end test, check that a dicomnode reject unknown AE titles"""

# Python standard library
from logging import DEBUG, INFO
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
from pynetdicom.ae import ApplicationEntity
# They are generating the list at run time so static checker can't handle this
from pynetdicom.sop_class import Verification # type: ignore

# Dicomnode Modules
from dicomnode.constants import DICOMNODE_LOGGER_NAME, DICOMNODE_PROCESS_LOGGER
from dicomnode.server.input import AbstractInput
from dicomnode.server.nodes import AbstractPipeline

# Test packages
from tests.helpers import clear_logger
from tests.helpers.dicomnode_test_case import DicomnodeTestCase
from tests.helpers.inputs import NeverValidatingInput


#region Setup
ACCEPTED_AE_TITLE = "AE_TITLE_!"

class RejectionAETitle(AbstractPipeline):
  input = {"BAH" : NeverValidatingInput}
  ae_title = "REJECT"
  #pynetdicom_logger_level = INFO
  log_output=None
  require_called_aet = True
  require_calling_aet = [ACCEPTED_AE_TITLE]

transfer_syntax = [
  ExplicitVRLittleEndian,
  ImplicitVRLittleEndian,
  DeflatedExplicitVRLittleEndian,
  ExplicitVRBigEndian,
]


#region TestCase
class RejectionTestCase(DicomnodeTestCase):
  def setUp(self):
    self.node = RejectionAETitle()
    self.port = randint(1025,65535)
    self.node.port = self.port


  def tearDown(self) -> None:
    clear_logger(DICOMNODE_LOGGER_NAME)
    clear_logger(DICOMNODE_PROCESS_LOGGER)

    self.node.close()
    super().tearDown()

  def test_rejection(self):
    with self.assertLogs(self.node.logger) as cm:
      self.node.open(blocking=False)
    ae = ApplicationEntity(ae_title="NOT_KNOWN")
    self.assertRegexIn("Starting Server at address: * and AE: REJECT", cm.output)
    sleep(0.005)
    ae.add_requested_context(Verification, transfer_syntax)
    with self.assertLogs('dicomnode', DEBUG) as recorded_logs:
      assoc = ae.associate('127.0.0.1', self.port, ae_title="NOT TARGET")
      self.assertFalse(assoc.is_established)
    self.assertIn(f'DEBUG:dicomnode:Connection NOT_KNOWN rejected a connection', recorded_logs.output)

    with self.assertLogs('dicomnode', DEBUG) as recorded_logs:
      assoc = ae.associate('127.0.0.1', self.port, ae_title="REJECT")
      self.assertFalse(assoc.is_established)
    self.assertIn(f'DEBUG:dicomnode:Connection NOT_KNOWN rejected a connection', recorded_logs.output)

    ae.ae_title = ACCEPTED_AE_TITLE
    with self.assertLogs('dicomnode', DEBUG) as recorded_logs:
      assoc = ae.associate('127.0.0.1', self.port, ae_title="NOT TARGET")
      self.assertFalse(assoc.is_established)
    self.assertIn(f'DEBUG:dicomnode:Connection {ACCEPTED_AE_TITLE} rejected a connection', recorded_logs.output)

    with self.assertLogs(self.node.logger, DEBUG) as recorded_logs:
      assoc = ae.associate('127.0.0.1', self.port, ae_title="REJECT")
      self.assertTrue(assoc.is_established)
      responds = assoc.send_c_echo()
      self.assertEqual(responds.Status, 0x0000)
      assoc.release()
    self.assertIn(f'DEBUG:dicomnode:Connection {ACCEPTED_AE_TITLE} send an echo', recorded_logs.output)
