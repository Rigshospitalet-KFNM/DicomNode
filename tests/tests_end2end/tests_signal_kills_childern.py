"""This is a test case of a historic input"""

# Python3 standard library
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

    test_port = randint(1024, 45000)

    def process_function():
      class DumbInput(AbstractInput):
        def validate(self) -> bool:
          return True

      class LoggingPipeline(AbstractPipeline):
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
          sleep(10)
          return NoOutput()

      instance = LoggingPipeline()
      instance.open(blocking=True)

    victim_process = Process(target=process_function)
    victim_process.start()

    victim_pid = victim_process.pid
    if victim_pid is None:
      raise AssertionError("This should not happen!")

    dataset = Dataset()
    dataset.SOPInstanceUID = gen_uid()
    dataset.SOPClassUID = SecondaryCaptureImageStorage
    dataset.PatientID = "Patient^ID"
    dataset.InstanceNumber = 1
    make_meta(dataset)

    # Need a small delay
    sleep(0.3)


    send_images("SENDER", Address('127.0.0.1', test_port, "TEST"), [dataset])

    # Wait for processing to begin. Although I am not sure why there's 2?
    while len(PS_Process(victim_process.pid).children(True)) >= 2:
      pass


    kill(victim_pid, signal.SIGINT)
    # Sadly we need a small delay here for the subprocesses to handle the signal
    # and die
    sleep(0.5)

    passes = True

    for process in PS_Process(victim_process.pid).children(True):
      if process.is_running():
        print("Had to manually kill victim's children")
        process.kill()
        passes = False

    if victim_process.is_alive():
      passes = False
      print("Had to manually kill Victim")
      victim_process.kill()

    self.assertTrue(passes)
