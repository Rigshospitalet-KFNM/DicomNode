# Python Standard library
import logging
from logging.handlers import TimedRotatingFileHandler
import os
from pathlib import Path
import sys
from typing import TextIO

# Third party modules

# Dicomnode Modules
from dicomnode.constants import DICOMNODE_PROCESS_LOGGER, DICOMNODE_LOGGER_NAME
from dicomnode.config import DicomnodeConfig, config_from_raw
from dicomnode.lib.io import TemporaryWorkingDirectory
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

  def test_exploration_rotating_log_handler_on_both(self):
    """Test that show case the behavior of loggers with multiple TimedRotatedFileHandlers"""
    logger_1 = logging.getLogger("Test-Logger-1")
    logger_2 = logging.getLogger("Test-Logger-2")

    logger_1.handlers.clear()
    logger_2.handlers.clear()

    # If you ever make python truly multithreaded this test suite is gonna be
    # sad
    with TemporaryWorkingDirectory(f"{self._testMethodName}") as tmp:
      file_name = "logout.log"

      # Note that we create two handlers here to mimic production
      handler_1 = TimedRotatingFileHandler(filename=file_name, backupCount=2)
      handler_2 = TimedRotatingFileHandler(filename=file_name, backupCount=2)

      logger_1.addHandler(handler_1)
      logger_2.addHandler(handler_2)

      logger_1.setLevel(logging.INFO)
      logger_2.setLevel(logging.INFO)

      logger_1.propagate = False
      logger_2.propagate = False

      logger_1.info("Hello from logger 1")
      logger_2.info("Hello from logger 2")

      handler_1.doRollover()

      logger_1.info("Hello from logger 1")
      logger_2.info("Hello from logger 2")

      with open(file_name, 'r') as fp:
        text = fp.readlines()

      paths = []

      for path in Path(os.getcwd()).glob('*'):
        paths.append(path)

      self.assertEqual(len(paths), 2)
      self.assertEqual(len(text), 1)

      logger_1.handlers.clear()
      logger_2.handlers.clear()


  def test_exploration_rotating_log_same_handler_on_both(self):
    """Test that show case the behavior of loggers with multiple TimedRotatedFileHandlers"""
    logger_1 = logging.getLogger("Test-Logger-1")
    logger_2 = logging.getLogger("Test-Logger-2")

    with TemporaryWorkingDirectory(f"{self._testMethodName}") as tmp:
      file_path = "logout.log"

      paths = []

      for path in Path(os.getcwd()).glob('*'):
        paths.append(path)

      handler = TimedRotatingFileHandler(filename=file_path, backupCount=2)

      logger_1.handlers.clear()
      logger_2.handlers.clear()

      logger_1.addHandler(handler)
      logger_2.addHandler(handler)

      logger_1.setLevel(logging.INFO)
      logger_2.setLevel(logging.INFO)

      logger_1.propagate = False
      logger_2.propagate = False

      logger_1.info("Hello from logger 1")
      logger_2.info("Hello from logger 2")

      handler.doRollover()

      logger_1.info("Hello from logger 1")
      logger_2.info("Hello from logger 2")

      with Path(file_path).open('r') as fp:
        text = fp.readlines()

      paths = []

      for path in Path(os.getcwd()).glob('*'):
        paths.append(path)

      self.assertEqual(len(paths), 2)
      self.assertEqual(len(text), 2)

      logger_1.handlers.clear()
      logger_2.handlers.clear()
