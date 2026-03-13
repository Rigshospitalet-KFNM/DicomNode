"""This test case is to show that multiple threads can send images to a server
and it handles it correctly
"""

# Python3 standard library
import logging
from random import randint
import threading
from time import sleep
from typing import List, Optional
from unittest import TestCase

# Third party Packages
from pydicom import Dataset

# Dicomnode Packages
from dicomnode.constants import DICOMNODE_PROCESS_LOGGER, DICOMNODE_LOGGER_NAME
from dicomnode.data_structures.image_tree import DicomTree
from dicomnode.dicom import gen_uid
from dicomnode.dicom.dimse import Address, send_images_thread, send_images
from dicomnode.lib.parallelism import ProcessLikeThread
from dicomnode.server.pipeline_tree import InputContainer
from dicomnode.server.nodes import AbstractPipeline
from dicomnode.server.dicomnode_config import DicomnodeConfig, config_from_raw
from dicomnode.server.input import AbstractInput
from dicomnode.server.grinders import ListGrinder
from dicomnode.server.output import DicomOutput, PipelineOutput
from dicomnode.server.processor import AbstractProcessor

# Test packages
from tests.helpers import generate_numpy_datasets, personify, clear_logger, process_thread_check_leak
from tests.helpers.storage_endpoint import ENDPOINT_AE_TITLE, ENDPOINT_PORT,\
  TestStorageEndpoint
from tests.helpers.dicomnode_test_case import DicomnodeTestCase

INPUT_KW = "input"
TEST_AE_TITLE = "TEST_AE"
SENDER_AE_TITLE = "SENDER_AE"

class ConcurrentRunner(AbstractProcessor):
  def process(self, input_container: InputContainer) -> PipelineOutput:
    image_list = input_container[INPUT_KW]
    self.logger.info(f"Processing {len(image_list)}")

    return DicomOutput([(Address('127.0.0.1', ENDPOINT_PORT, ENDPOINT_AE_TITLE), image_list)], ENDPOINT_AE_TITLE)


class ListInput(AbstractInput):
  required_tags = [0x0008_0018]

  image_grinder = ListGrinder()

  def validate(self) -> bool:
    return True


class ConcurrencyNode(AbstractPipeline):
  """The main purpose of this pipeline is to have multiple associations
  And have them send data into the same input object at the same time.

  The Goal should be that the process functions runs once with all the pictures.

  This is mainly to showcase that file storage node support multiple associations.
  """

  patient_identifier_tag = 0x0020_000E
  ae_title = TEST_AE_TITLE
  input = { INPUT_KW : ListInput }
  require_calling_aet = [SENDER_AE_TITLE]
  log_level: int = logging.DEBUG
  log_output = None
  images_processed: Optional[int] = None
  processing_directory = None
  Processor = ConcurrentRunner


class ConcurrencyTestCase(DicomnodeTestCase):
  """These test are similar to the concurrent test earlier but with focus on
     some production issues from multiple assocation sending to the same
     input"""

  @process_thread_check_leak
  def test_spam_to_same_input(self):
    # This test is kinda difficult
    # Thread 1
    #    ¦     -> Node -> TestStorageEndpoint
    # Thread N


    with self.assertLogs(DICOMNODE_LOGGER_NAME):
      self.node = ConcurrencyNode()
      self.test_port = randint(1025, 49999)
      self.node.port = self.test_port
      self.node.open(blocking=False)

      self.endpoint = TestStorageEndpoint()
      self.endpoint.open()
      with self.assertNonCapturingLogs(DICOMNODE_PROCESS_LOGGER):
        num_threads = 6
        num_images = 5

        address = Address('localhost', self.test_port, TEST_AE_TITLE)
        patient_cpr = "0201919996"

        sender_threads: List[threading.Thread] = []

        study_uid = gen_uid()
        series_uid = gen_uid()
        for i in range(num_threads):
            images = DicomTree(generate_numpy_datasets(num_images,
                                                        PatientID=patient_cpr,
                                                        SeriesUID=series_uid,
                                                        StudyUID=study_uid,
                                                        Rows=10,
                                                        Cols=10,
                                                        ))
            images.map(personify(
              tags=[
                (0x00100010, "PN", "Odd Name Test"),
                (0x00100040, "CS", "M")
              ]
            ))

            thread = ProcessLikeThread(
              name=f"sending-thread-{i + 1}",
              target=send_images, args=[SENDER_AE_TITLE, address, images])
            thread.start()
            sender_threads.append(thread)

        [thread.join() for thread in sender_threads]
        [thread.join() for thread in self.node.dicom_application_entry.active_associations]
        [thread.join() for thread in self.endpoint.ae.active_associations]


        self.node.close()
        self.endpoint.close()

        self.assertEqual(len(self.endpoint.storage[patient_cpr]), num_threads * num_images)
        self.assertEqual(self.endpoint.accepted_associations, 1)

    clear_logger(DICOMNODE_LOGGER_NAME)
    clear_logger(DICOMNODE_PROCESS_LOGGER)

  def test_inputs_are_thread_safe_append_to(self):
    num_threads = 60
    num_datasets_per_thread = 5

    config = config_from_raw()

    input_ = ListInput(config)

    def generate_dataset():
      ds = Dataset()
      ds.SOPInstanceUID = gen_uid()
      return ds

    def thread_target():
      for i in range(num_datasets_per_thread):
        input_.add_image(generate_dataset())

    threads = [
      ProcessLikeThread(
        target=thread_target,
        name=f"Spammer-{i+1}"
      ) for i in range( num_threads)
    ]

    [thread.start() for thread in threads]
    [thread.join() for thread in threads]

    self.assertEqual(input_.images, num_threads * num_datasets_per_thread)
