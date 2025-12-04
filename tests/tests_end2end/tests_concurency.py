"""This test case is to show that multiple threads can send images to a server
and it handles it correctly
"""

# Python3 standard library
import logging
from random import randint
import threading
from typing import List, Optional
from unittest import TestCase

# Third party Packages

# Dicomnode Packages
from dicomnode.data_structures.image_tree import DicomTree
from dicomnode.dicom import gen_uid
from dicomnode.dicom.dimse import Address, send_images_thread
from dicomnode.server.pipeline_tree import InputContainer
from dicomnode.server.nodes import AbstractPipeline
from dicomnode.server.output import DicomOutput, PipelineOutput

# Test packages
from tests.helpers import generate_numpy_datasets, personify
from tests.helpers.inputs import ListInput
from tests.helpers.storage_endpoint import ENDPOINT_AE_TITLE, ENDPOINT_PORT,\
  TestStorageEndpoint
from tests.helpers.dicomnode_test_case import DicomnodeTestCase

INPUT_KW = "input"
TEST_AE_TITLE = "TEST_AE"
SENDER_AE_TITLE = "SENDER_AE"


class ConcurrencyNode(AbstractPipeline):
  """The main purpose of this pipeline is to have multiple associations
  And have them send data into the same input object at the same time.

  The Goal should be that the process functions runs once with all the pictures.

  This is mainly to showcase that file storage node support multiple associations.
  """

  patient_identifier_tag = 0x0020_000E
  ae_title = TEST_AE_TITLE
  input = {INPUT_KW : ListInput }
  require_calling_aet = [SENDER_AE_TITLE]
  log_level: int = logging.DEBUG
  log_output = None
  images_processed: Optional[int] = None
  processing_directory = None


  def process(self, input_data: InputContainer) -> PipelineOutput:
    image_list = input_data[INPUT_KW]

    self.images_processed = len(image_list)

    return DicomOutput([(Address('127.0.0.1', ENDPOINT_PORT, ENDPOINT_AE_TITLE), image_list)], self.ae_title)


class ConcurrencyTestCase(DicomnodeTestCase):
  """These test are similar to the concurrent test earlier but with focus on
     some production issues from multiple assocation sending to the same
     input"""
  def setUp(self):
    self.node = ConcurrencyNode()
    self.test_port = randint(1025, 49999)
    self.node.port = self.test_port
    self.node.open(blocking=False)

  def tearDown(self) -> None:
    self.node.close()


  def performance_spam_to_same_input(self):
    # This test is kinda difficult
    # Thread 1
    #    ¦     -> Node -> TestStorageEndpoint
    # Thread N


    release_event = threading.Event()
    self.endpoint = TestStorageEndpoint(release_event=release_event)
    self.endpoint.open()

    num_threads = 3
    num_images = 50

    address = Address('localhost', self.test_port, TEST_AE_TITLE)
    patient_cpr = "0201919996"

    sender_threads: List[threading.Thread] = []

    study_uid = gen_uid()
    series_uid = gen_uid()
    for _ in range(num_threads):
        images = DicomTree(generate_numpy_datasets(num_images,
                                                   PatientID=patient_cpr,
                                                   SeriesUID=series_uid,
                                                   StudyUID=study_uid,
                                                   ))
        images.map(personify(
          tags=[
            (0x00100010, "PN", "Odd Name Test"),
            (0x00100040, "CS", "M")
          ]
        ))

        thread = send_images_thread(SENDER_AE_TITLE, address, images, None, False)
        sender_threads.append(thread)

    [thread.join() for thread in sender_threads]
    [thread.join() for thread in self.node.dicom_application_entry.active_associations]
    [thread.join() for thread in self.endpoint.ae.active_associations]

    self.assertEqual(len(self.endpoint.storage[patient_cpr]), num_threads * num_images)
    self.assertEqual(self.endpoint.accepted_associations, 1)

    self.endpoint.close()
