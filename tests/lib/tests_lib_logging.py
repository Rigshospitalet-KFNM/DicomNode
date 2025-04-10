# Python standard library
import logging
from logging.handlers import QueueHandler
import multiprocessing
from multiprocessing import Process
from random import random
import time
from multiprocessing import Queue
from typing import List

# Dicomnode test
from dicomnode.lib import logging as dnl

# Test helpers
from tests.helpers.dicomnode_test_case import DicomnodeTestCase

def worker(queue: Queue):
  handler = QueueHandler(queue)
  root = logging.getLogger()
  root.addHandler(handler)
  name = multiprocessing.current_process().name

  for i in range(10):
    time.sleep(random() / 1000)
    logger = logging.getLogger()
    message = f"Logger {name}: message {i}"
    #print(message)
    logger.info(message)


class LoggingTestcase(DicomnodeTestCase):
  def test_logging(self):
    queue = Queue()

    listener = Process(
      target=dnl.listener_logger,
      args=(queue,)
    )
    listener.start()

    workers: List[Process] = []
    for i in range(10):
      worker_process = Process(
        target=worker(queue),
        name=f"Process {i + 1}"
      )
      worker_process.start()

      workers.append(worker_process)

    for w in workers:
      w.join()

    queue.put_nowait(None)
