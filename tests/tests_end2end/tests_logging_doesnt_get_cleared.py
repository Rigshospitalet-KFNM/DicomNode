# Python standard library
import sys


# Third party library

# Dicomnode modules
from dicomnode.constants import DICOMNODE_LOGGER_NAME
from dicomnode.lib.logging import LoggerConfig, get_logger, set_logger
from dicomnode.server.input_container import InputContainer
from dicomnode.server.output import PipelineOutput, NoOutput
from dicomnode.server.processor import AbstractProcessor, ProcessRunnerArgs

# Tests helpers
from tests.helpers import clear_logger
from tests.helpers.dicomnode_test_case import DicomnodeTestCase

class Processor(AbstractProcessor):
  def process(self, input_container: InputContainer) -> PipelineOutput:
    self.logger.info(f"My loggers are: {self.logger.handlers}")
    return NoOutput()



class End2EndLogging(DicomnodeTestCase):
  def test_logging_does_not_get_cleared(self):
    logger = get_logger()

    logging_config = LoggerConfig(
      log_output=None
    )

    set_logger(logger, logging_config)

    input_container = InputContainer({},{})

    self.assertEqual(len(logger.handlers), 1)
    handlers_before = logger.handlers[0]


    args = ProcessRunnerArgs(
      input_container, logging_config, None, "Tests"
    )

    Processor(args)

    self.assertEqual(len(logger.handlers), 1)
    handlers_after = logger.handlers[0]

    Processor(args)

    self.assertEqual(len(logger.handlers), 1)
    handlers_after_after = logger.handlers[0]

    self.assertIsSameType(handlers_after, handlers_before)
    self.assertIsSameType(handlers_after_after, handlers_before)

    clear_logger(DICOMNODE_LOGGER_NAME)
