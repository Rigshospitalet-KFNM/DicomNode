# Python standard library
import sys
from io import StringIO
from random import randint

# Third party library

# Dicomnode modules
from dicomnode.constants import DICOMNODE_LOGGER_NAME
from dicomnode.config import config_from_raw
from dicomnode.lib.logging import LoggerConfig, get_logger, set_logger
from dicomnode.server.input_container import InputContainer
from dicomnode.server.output import PipelineOutput, NoOutput
from dicomnode.server.processor import AbstractProcessor, ProcessRunnerArgs
from dicomnode.server.patient_node import PatientNode
from dicomnode.server.nodes import AbstractPipeline
from dicomnode.server.input import AbstractInput

# Tests helpers
from tests.helpers import clear_logger
from tests.helpers.dicomnode_test_case import DicomnodeTestCase

class Processor_(AbstractProcessor):
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

    Processor_(args)

    self.assertEqual(len(logger.handlers), 1)
    handlers_after = logger.handlers[0]

    Processor_(args)

    self.assertEqual(len(logger.handlers), 1)
    handlers_after_after = logger.handlers[0]

    self.assertIsSameType(handlers_after, handlers_before)
    self.assertIsSameType(handlers_after_after, handlers_before)

    clear_logger(DICOMNODE_LOGGER_NAME)


  def test_logging_does_not_get_cleared_stdout(self):
    logger = get_logger()

    pipe = StringIO()

    logging_config = LoggerConfig(
      log_output=pipe
    )

    set_logger(logger, logging_config)

    input_container = InputContainer({},{})

    self.assertEqual(len(logger.handlers), 1)
    handlers_before = logger.handlers[0]


    args = ProcessRunnerArgs(
      input_container, logging_config, None, "Tests"
    )

    Processor_(args)

    self.assertEqual(len(logger.handlers), 1)
    handlers_after = logger.handlers[0]

    Processor_(args)

    self.assertEqual(len(logger.handlers), 1)
    handlers_after_after = logger.handlers[0]

    self.assertIsSameType(handlers_after, handlers_before)
    self.assertIsSameType(handlers_after_after, handlers_before)

    clear_logger(DICOMNODE_LOGGER_NAME)

  def test_pipeline_node_fucks_with_things(self):
    port_ = randint(1050, 45000)

    output = StringIO()

    class Input(AbstractInput):
      def validate(self) -> bool:
        return True

    class Pipeline(AbstractPipeline):
      Processor = Processor_

      port = port_

      input = {
        "INPUT" : Input
      }

      log_output = output


    pipeline = Pipeline()

    handlers_before = pipeline.logger.handlers[0]

    node = PatientNode("HelloWorld", {}, config_from_raw())

    pipeline._process_output([("HelloWorld",node)], None)

    handlers_after = pipeline.logger.handlers[0]

    self.assertIsSameType(handlers_before,handlers_after)

    output.seek(0)
    string_output = output.read()
    self.assertIn("process HelloWorld successful", string_output)

    clear_logger(DICOMNODE_LOGGER_NAME)
