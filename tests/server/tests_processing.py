# Python standard library
import os
from pathlib import Path
from unittest.mock import patch
from typing import Any, List, Tuple

# Third party modules

# Dicomnode Modules
from dicomnode.constants import DICOMNODE_LOGGER_NAME
from dicomnode.data_structures.optional import OptionalPath
from dicomnode.lib.logging import LoggerConfig
from dicomnode.server.input_container import InputContainer
from dicomnode.server.output import PipelineOutput, NoOutput
from dicomnode.server.processor import AbstractProcessor, ProcessRunnerArgs

# Tests Modules
from tests.helpers.dicomnode_test_case import DicomnodeTestCase

class RaisingOutput(PipelineOutput):
  def __init__(self, output: List[Tuple[Any, Any]]) -> None:
    pass

  def send(self) -> bool:
    raise Exception("HELLO I'm the problem")

class DummyProcessor(AbstractProcessor):
  def process(self, input_container: InputContainer) -> PipelineOutput:
    return RaisingOutput([])

class RaisingProcessor(AbstractProcessor):
  def process(self, input_container: InputContainer) -> PipelineOutput:
    raise Exception("HELLO I'm the problem")

class LoggingProcessor(AbstractProcessor):
  def process(self, input_container: InputContainer) -> PipelineOutput:
    self.logger.info(f"MY CWD IS: {os.getcwd()}")

    return NoOutput()

class ProcessorTestCase(DicomnodeTestCase):
  def test_processor_logs_dispatch_exception(self):
    args = ProcessRunnerArgs(
      InputContainer({},{},OptionalPath()),
      LoggerConfig(log_output=None),
      None, "Test"
    )

    with patch('dicomnode.server.processor.set_logger'):
      with self.assertLogs(DICOMNODE_LOGGER_NAME) as captured_logs:
        DummyProcessor(args)

    self.assertRegexIn("HELLO I'm the problem", captured_logs.output)
    self.assertRegexIn("unable to dispatch", captured_logs.output)

  def test_processor_logs_processor_exception(self):
    args = ProcessRunnerArgs(
      InputContainer({},{},OptionalPath()),
      LoggerConfig(log_output=None),
      None, "Test"
    )

    with patch('dicomnode.server.processor.set_logger'):
      with self.assertLogs(DICOMNODE_LOGGER_NAME) as captured_logs:
        RaisingProcessor(args)

    self.assertRegexIn("HELLO I'm the problem", captured_logs.output)
    self.assertRegexIn("in process function", captured_logs.output)

  def test_processor_log_using_temp_working_directory(self):
    args = ProcessRunnerArgs(
      InputContainer({},{},OptionalPath()),
      LoggerConfig(log_output=None),
      Path(self._testMethodName), "Test"
    )

    before_cwd = os.getcwd()

    with patch('dicomnode.server.processor.set_logger'):
      with self.assertLogs(DICOMNODE_LOGGER_NAME) as captured_logs:
        LoggingProcessor(args)

    after_cwd = os.getcwd()

    self.assertEqual(before_cwd, after_cwd)

    self.assertRegexIn(f"{os.getcwd()}/{self._testMethodName}" ,captured_logs.output)
    self.assertFalse(Path(self._testMethodName).exists())