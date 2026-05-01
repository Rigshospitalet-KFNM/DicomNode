"""Test cases for dicomnode.server.nodes without any networking in them."""

# Python standard library
from logging import getLogger
from logging import NullHandler
from unittest.mock import patch, MagicMock

# Third party modules
from pydicom import Dataset

# Dicomnode modules
from dicomnode.constants import DICOMNODE_LOGGER_NAME
from dicomnode.dicom import gen_uid
from dicomnode.server.nodes import AbstractPipeline
from dicomnode.server.processor import AbstractProcessor

# Test helpers
from tests.helpers import clear_logger
from tests.helpers.dicomnode_test_case import DicomnodeTestCase

class Processor(AbstractProcessor):
  pass

class Node(AbstractPipeline):
  Processor = Processor
  log_output = None

  port = 0


class NonNetworkNodeTests(DicomnodeTestCase):
  def tearDown(self) -> None:
    clear_logger(DICOMNODE_LOGGER_NAME)
    return super().tearDown()

  def test_node_logs_to_dicomnode_logger(self):
    with patch('dicomnode.lib.logging.set_logger'):

      node = Node()

      with self.assertLogs(DICOMNODE_LOGGER_NAME) as captured_logs:
        node.logger.info("HELLO WORLD")

      self.assertRegexIn("HELLO WORLD", captured_logs.output)

  def test_node_logger_output_none_gives_null_handler(self):
    node = Node()
    self.assertIsInstance(node.logger.handlers[0], NullHandler)


  def test_node_opening_an_open_node_and_close_and_closing(self):
    node = Node()

    with self.assertLogs(DICOMNODE_LOGGER_NAME) as captured_logs:
      node.close()

    self.assertRegexIn("Attempted to close an closed node", captured_logs.output)

    with self.assertLogs(DICOMNODE_LOGGER_NAME) as captured_logs_2:
      node.open(blocking=False)
      node.open(blocking=False)

      node.close()

    self.assertRegexIn("Attempted to open a node, that was already open", captured_logs_2.output)


  def test_node_handle_close_logs_failed_datasets(self):
    node = Node()

    event_mock = MagicMock()

    event_mock.address = ['address']
    event_mock.assoc.requestor.ae_title = "FUCK"

    with self.assertLogs(DICOMNODE_LOGGER_NAME):
      with patch('dicomnode.server.pipeline_storage.ReactivePipelineStorage.extract_input_container') as extract_mock:
        dataset = Dataset()
        dataset.PatientID = "pid"
        dataset.StudyInstanceUID = gen_uid()
        dataset.SeriesInstanceUID = gen_uid()
        dataset.SOPInstanceUID = gen_uid()

        dataset.StudyDescription = "a study description"
        dataset.SeriesDescription = "\"a series description\""

        extract_mock.return_value = ([],[dataset])

        node._handle_connection_closed(event_mock)