"""Test cases for dicomnode.server.nodes without any networking in them."""

# Python standard library
from logging import getLogger
from logging import NullHandler
from unittest.mock import patch

# Third party modules

# Dicomnode modules
from dicomnode.constants import DICOMNODE_LOGGER_NAME
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


class NonNetworkNodeTests(DicomnodeTestCase):
  def test_node_logs_to_dicomnode_logger(self):
    with patch('dicomnode.lib.logging.set_logger'):

      node = Node()

      with self.assertLogs(DICOMNODE_LOGGER_NAME) as captured_logs:
        node.logger.info("HELLO WORLD")

      self.assertRegexIn("HELLO WORLD", captured_logs.output)

  def test_node_logger_output_none_gives_null_handler(self):
    node = Node()

    self.assertIsInstance(node.logger.handlers[0], NullHandler)

    clear_logger(DICOMNODE_LOGGER_NAME)
