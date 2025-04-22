"""This is a test case of a historic input"""

# Python3 standard library
from logging import DEBUG
import signal
from os import kill, getpid
from time import sleep
from random import randint
from multiprocessing import Process

# Third party Packages
from pydicom import Dataset
from pydicom.uid import SecondaryCaptureImageStorage
from psutil import Process as PS_Process

# Dicomnode Packages
from dicomnode.constants import DICOMNODE_PROCESS_LOGGER
from dicomnode.dicom import make_meta, gen_uid
from dicomnode.dicom.dimse import Address, send_images
from dicomnode.server.nodes import AbstractPipeline
from dicomnode.server.input import AbstractInput
from dicomnode.server.output import PipelineOutput, NoOutput

# Testing Packages
from helpers.dicomnode_test_case import DicomnodeTestCase

class SignalKillsChildrenWritten(DicomnodeTestCase):
  def test_signal_kills_children(self):
    "This test checks if you send a SIGINT / SIGTERM / SIGKILL to the node"\
    " process, that it kills any running child processes. Note that this"\
    " test is flaky, and you might need increase delays"

    # I will also note there's a flake, here where some times the signal doesn't
    # propagate. I have no idea why. An alternative would perhaps be you create
    # a shared flag with a thread, that repeatable checks it, and if signaled
    # Just calls exit

    start_up_delay = 0.2

    test_port = randint(1024, 45000)


    def process_function():
      class DumbInput(AbstractInput):
        def validate(self) -> bool:
          return True

      class DummyPipeline(AbstractPipeline):
        port = test_port
        ae_title = "TEST"

        input = {
          "input" : DumbInput
        }

        def __init__(self, config=None) -> None:
          super().__init__(config)
          self.logger.info(f"Pipeline process pid: {getpid()}")

        log_output = None

        def process(self, input_data):
          self.logger.info(f"Processing Process {getpid()} starting")
          sleep(10)
          return NoOutput()

      instance = DummyPipeline()
      instance.open(blocking=True)

    victim_process = Process(target=process_function)
    victim_process.start()

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

    for process in PS_Process(victim_process.pid).children(True):
      while process.is_running():
        pass

    while victim_process.is_alive():
      pass
