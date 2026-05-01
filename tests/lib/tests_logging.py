# Python Standard library
import sys
import logging
from typing import TextIO

# Third party modules

# Dicomnode Modules
from dicomnode.constants import DICOMNODE_PROCESS_LOGGER, DICOMNODE_LOGGER_NAME
from dicomnode.config import DicomnodeConfig, config_from_raw
from dicomnode.lib.logging import set_logger, LogManager, LoggerConfig, get_logger

# Tests Helper functions
from tests.helpers import clear_logger
from tests.helpers.dicomnode_test_case import DicomnodeTestCase

class LoggingTests(DicomnodeTestCase):
  def test_logger_config_that_produces_stdout(self):
    root_logger = logging.getLogger()

    config = config_from_raw()
    config.LOG_OUTPUT = "stdout"

    manager = LogManager(config, None)

    self.assertTrue(sys.stdout, TextIO)

    logger = get_logger()

    self.assertEqual(len(logger.handlers), 1)
    self.assertIsInstance(logger.handlers[0], logging.StreamHandler)

    logger.handlers.clear()