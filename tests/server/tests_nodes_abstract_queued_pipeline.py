import inspect
import threading
import multiprocessing
from time import sleep
from typing import List

from pydicom import DataElement, Dataset
from pynetdicom.association import Association
from pynetdicom import events as evt

from tests.helpers import generate_numpy_datasets, clear_logger
from tests.helpers.dicomnode_test_case import DicomnodeTestCase

from dicomnode.constants import DICOMNODE_LOGGER_NAME, DICOMNODE_PROCESS_LOGGER
from dicomnode.dicom.series import DicomSeries
from dicomnode.lib.logging import set_logger
from dicomnode.server.factories.association_events import ReleasedEvent, AssociationTypes
from dicomnode.server.grinders import ListGrinder
from dicomnode.server.input import AbstractInput
from dicomnode.server.nodes import AbstractQueuedPipeline
from dicomnode.server.output import NoOutput, PipelineOutput
from dicomnode.server.pipeline_tree import InputContainer
from dicomnode.server.process_runner import Processor

class TestInput(AbstractInput):
  def validate(self) -> bool:
    return self.images > 0
  image_grinder = ListGrinder()

INPUT_KW = "dicoms"

PATIENT_ID_1 = "Series 1"
PATIENT_ID_2 = "Series 2"
PATIENT_ID_3 = "Series 3"


series_1 = DicomSeries([ds for ds in generate_numpy_datasets(10, Cols=10, Rows=10, PatientID=PATIENT_ID_1)])
series_2 = DicomSeries([ds for ds in generate_numpy_datasets(10, Cols=10, Rows=10, PatientID=PATIENT_ID_2)])
series_3 = DicomSeries([ds for ds in generate_numpy_datasets(10, Cols=10, Rows=10, PatientID=PATIENT_ID_3)])

#
series_1[0x0011_0100] = DataElement(0x0011_0100, 'FT', 0.050)
series_2[0x0011_0100] = DataElement(0x0011_0100, 'FT', 0.015)
series_3[0x0011_0100] = DataElement(0x0011_0100, 'FT', 0.005)

class AssociationDummy():
  class Helper:
    class Helper2:
      abstract_syntax = "1.2.840.10008.5.1.4.1.1"

    ae_title = "dummy ae title"
    address = "dummy address"
    requested_contexts = [
      Helper2()
    ]

  requestor = Helper() #type: ignore

  def __init__(self, thread_id):
    self.native_id = thread_id


class TestProcessRunner(Processor):
  def process(self, input_container: InputContainer) -> PipelineOutput:
    datasets: List[Dataset] = input_container["dicoms"]

    pivot = datasets[0]
    sleep(pivot[0x0011_0100].value)
    self.logger.info(f"Processed {pivot.PatientID}")

    return NoOutput()

class TestPipeline(AbstractQueuedPipeline):
  port = 51251
  input = {
    "dicoms" : TestInput
  }
  log_output=None

  process_runner = TestProcessRunner


class QueuedPipelineTestCase(DicomnodeTestCase):
  def tearDown(self) -> None:
    clear_logger(DICOMNODE_PROCESS_LOGGER)
    clear_logger(DICOMNODE_LOGGER_NAME)
    super().tearDown()

  def test_queue_runner_has_the_correct_handler(self):
    node = TestPipeline()

    self.assertEqual(inspect.getsource(node._handle_association_released),
                     inspect.getsource(node._evt_handlers[evt.EVT_RELEASED]))

    self.assertEqual([t for t in threading.enumerate()], [threading.main_thread()])
    self.assertEqual([process for process in multiprocessing.get_context('spawn').active_children()], [])


  def test_queue_pipeline(self):
    node = TestPipeline()

    thread_id_1 = 390
    thread_id_2 = 392
    thread_id_3 = 395

    node._updated_patients[thread_id_1] = { PATIENT_ID_1 : 0 }
    node._updated_patients[thread_id_2] = { PATIENT_ID_2 : 0 }
    node._updated_patients[thread_id_3] = { PATIENT_ID_3 : 0 }
    node._patient_locks[PATIENT_ID_1] = (set([thread_id_1]), threading.Lock())
    node._patient_locks[PATIENT_ID_2] = (set([thread_id_2]), threading.Lock())
    node._patient_locks[PATIENT_ID_3] = (set([thread_id_3]), threading.Lock())

    node.data_state.add_images(series_1)
    node.data_state.add_images(series_2)
    node.data_state.add_images(series_3)

    with self.assertLogs(DICOMNODE_LOGGER_NAME):
      node.open(blocking=False)
      with self.assertLogs(DICOMNODE_PROCESS_LOGGER):
        # Yeah I know, I should mock, but this is easier...
        event_1 = evt.Event(AssociationDummy(thread_id_1), evt.EVT_RELEASED) # type: ignore
        event_2 = evt.Event(AssociationDummy(thread_id_2), evt.EVT_RELEASED) # type: ignore
        event_3 = evt.Event(AssociationDummy(thread_id_3), evt.EVT_RELEASED) # type: ignore

        node._handle_association_released(event_1)
        node._handle_association_released(event_2)
        node._handle_association_released(event_3)
        node.process_queue.join()

        node.close()

    node.running = False
    node.process_queue.join()
    node.process_thread.join()

    self.assertEqual([t for t in threading.enumerate()], [threading.main_thread()])
    self.assertEqual([process for process in multiprocessing.get_context('spawn').active_children()], [])


    #self.assertRegexBefore(PATIENT_ID_1, PATIENT_ID_2, cm.output)
    #self.assertRegexBefore(PATIENT_ID_1, PATIENT_ID_3, cm.output)
    #self.assertRegexBefore(PATIENT_ID_2, PATIENT_ID_3, cm.output)