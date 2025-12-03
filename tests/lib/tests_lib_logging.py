# Python standard library
import logging
from logging.handlers import QueueHandler
import multiprocessing
from multiprocessing import Process
from random import random, randint
import time
from multiprocessing import Queue
from threading import Thread
from typing import List
from unittest import skip

# Third party modules
from pydicom import Dataset
from pydicom.uid import SecondaryCaptureImageStorage

# Dicomnode Modules
from dicomnode.dicom import gen_uid, make_meta
from dicomnode.dicom.dimse import send_images, Address

from dicomnode.lib import logging as dnl
from dicomnode.lib.utils import spawn_process, spawn_thread
from dicomnode.constants import DICOMNODE_LOGGER_NAME, DICOMNODE_PROCESS_LOGGER
from dicomnode.server.nodes import AbstractPipeline
from dicomnode.server.input import AbstractInput
from dicomnode.server.output import NoOutput

# Test helpers
from tests.helpers import process_thread_check_leak
from tests.helpers.dicomnode_test_case import DicomnodeTestCase

def worker(queue: Queue):
  root = logging.getLogger()
  root.handlers.clear()
  root.addHandler(QueueHandler(queue))
  root.setLevel(logging.DEBUG)

  name = multiprocessing.current_process().name
  pid = multiprocessing.current_process().pid

  for i in range(3):
    time.sleep(random() / 1000)
    logger = logging.getLogger()
    message = f"Logger {name} - {pid}: message {i}"
    logger.info(message)

  @process_thread_check_leak
  def test_sample_queue_logging(self):
    queue = Queue()
    listener = spawn_thread(
      dnl.listener_logger,
      queue
    )
    logger = logging.getLogger(DICOMNODE_PROCESS_LOGGER)

    with self.assertLogs(logger) as ctx:
      workers: List[Process] = []
      for i in range(3):
        worker_process = spawn_process(
          worker, queue, logger=logger, start=False
        )
        workers.append(worker_process)

      for w in workers:
        w.start()

      for w in workers:
        w.join()

      queue.put_nowait(None)
      listener.join()

  @process_thread_check_leak
  def test_queue_logging_end2end(self):
    test_port = randint(10250, 30000)

    class TestInput(AbstractInput):
      def validate(self) -> bool:
        return True

    class TestPipeline(AbstractPipeline):
      ae_title = "TEST"
      port = test_port

      input = {
        'TEST' : TestInput
      }

      def process(self, input_data):
        self.logger.critical("Hello from Test")
        self.logger.critical(self.logger.handlers)
        return NoOutput()

    instance = TestPipeline()
    instance.open(blocking=False)

    dataset = Dataset()
    dataset.SOPInstanceUID = gen_uid()
    dataset.SOPClassUID = SecondaryCaptureImageStorage
    dataset.PatientID = "Patient^ID"
    dataset.InstanceNumber = 1
    make_meta(dataset)

    # So this test is fucking stupid. Well problem is we wanna test that the
    # process logger gets the messages, but because they are chained together
    # We get emission from both, however if you capture both logs, then
    # the way the assertLogsContextManager works is by replacing the handlers
    # Which means the process logger doesn't get the message, so if you test
    # the thing you wanna test, you get nosy output, otherwise the you can't
    # test it. The solution is to write a none capturing LogContextManager
    # TODO: TODO: TODO:

    logger = logging.getLogger(DICOMNODE_PROCESS_LOGGER)

    with self.assertLogs(DICOMNODE_LOGGER_NAME):
      send_images("SENDER", Address('127.0.0.1', test_port, "TEST"), [dataset])

      instance.close()
