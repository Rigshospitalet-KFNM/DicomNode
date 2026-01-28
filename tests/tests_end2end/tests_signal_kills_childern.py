"""This is a test case of a historic input"""

# Python3 standard library
from logging import DEBUG, getLogger, LogRecord
import signal
from os import kill, getpid, getcwd
from time import sleep
import io
from multiprocessing import Process, Queue, get_context
from random import randint
from pathlib import Path
import sys
from typing import Any, Optional
from threading import Thread, get_native_id

# Third party Packages
from pydicom import Dataset
from pydicom.uid import SecondaryCaptureImageStorage
from psutil import Process as PS_Process

# Dicomnode Packages
from dicomnode.constants import DICOMNODE_PROCESS_LOGGER, DICOMNODE_LOGGER_NAME
from dicomnode.dicom import make_meta, gen_uid
from dicomnode.dicom.dimse import Address, send_images
from dicomnode.lib.utils import spawn_process
from dicomnode.lib import logging
from dicomnode.server.nodes import AbstractPipeline
from dicomnode.server.input import AbstractInput
from dicomnode.server.output import PipelineOutput, NoOutput
from dicomnode.server.pipeline_tree import InputContainer
from dicomnode.server.processor import AbstractProcessor

# Testing Packages
from tests.helpers import clear_logger
from tests.helpers.dicomnode_test_case import DicomnodeTestCase

class DumbProcessRunner(AbstractProcessor):
  def process(self, input_container: InputContainer) -> PipelineOutput:
    sleep(10.0)
    return super().process(input_container)

class DumbInput(AbstractInput):
          def validate(self) -> bool:
            return True

class DummyPipeline(AbstractPipeline):
  ae_title = "TEST"
  log_output = None

  input = {
    "input" : DumbInput
  }

  Processor = DumbProcessRunner


  def __init__(self, config=None, queue: Any=None) -> None:
    self.processing_directory = Path(getcwd()) / "processing"
    self._log_queue = queue
    super().__init__(config)
    self.logger.info(f"Pipeline process pid: {getpid()}")
    self.logger.info(f"Current handlers: {self.logger.handlers}")


def process_function(queue: Queue, test_port):
  instance = DummyPipeline(queue=queue)
  instance.port = test_port
  instance.open(blocking=True)


class SignalKillsChildrenWritten(DicomnodeTestCase):
  def tearDown(self) -> None:
    super().tearDown()

  def test_signal_kills_children(self):
    "This test checks if you send a SIGINT / SIGTERM / SIGKILL to the node"\
    " process, that it kills any running child processes. Note that this"\
    " test is flaky, and you might need increase delays"

    # I will also note there's a flake, here where some times the signal doesn't
    # propagate. I have no idea why. An alternative would perhaps be you create
    # a shared flag with a thread, that repeatable checks it, and if signaled
    # Just calls exit
    buf = io.StringIO()
    sys.stdout = buf
    logger = getLogger(DICOMNODE_LOGGER_NAME)

    logging_queue = get_context('spawn').Queue()
    start_up_delay = 2.0

    test_port = randint(1024, 45000)

    with self.assertLogs(DICOMNODE_LOGGER_NAME, level=DEBUG) as captured_logs:
      # So this line, somehow captures the logs from the other process, although
      # it doesn't register that it did indeed catch them

      victim_process = spawn_process(process_function, logging_queue, test_port)

      victim_pid = victim_process.pid
      if victim_pid is None:
        raise AssertionError("Failed to start the process")

      dataset = Dataset()
      dataset.SOPInstanceUID = gen_uid()
      dataset.SOPClassUID = SecondaryCaptureImageStorage
      dataset.PatientID = "Patient^ID"
      dataset.InstanceNumber = 1
      make_meta(dataset)

      # Need a small delay
      sleep(start_up_delay)

      send_images("SENDER", Address('127.0.0.1', test_port, "TEST"), [dataset])

      # Wait for processing to begin. Although I am not sure why there's 2?
      sleep(start_up_delay)

      kill(victim_pid, signal.SIGINT)

      process_is_alive_count = 0

      killed_successful = True

      for process in PS_Process(victim_process.pid).children(True):
        while process.is_running():
          if process_is_alive_count < 100:
            process_is_alive_count += 1
            sleep(0.01)
          else:
            killed_successful = False
            kill(process.pid, signal.SIGKILL)


      while victim_process.is_alive():
        if process_is_alive_count < 100:
          process_is_alive_count += 1
          sleep(0.01)
        else:
          killed_successful = False
          kill(victim_pid, signal.SIGKILL)

    logging_queue.put_nowait(None)
    logging_queue.close()

    if not killed_successful:
      raise AssertionError("Was unable to kill the processes!")
