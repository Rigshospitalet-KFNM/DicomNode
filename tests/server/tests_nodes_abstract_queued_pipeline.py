import inspect
import threading
from time import sleep
from typing import List

from pydicom import DataElement, Dataset
from pynetdicom.association import Association
from pynetdicom import events as evt

from tests.helpers import generate_numpy_datasets
from tests.helpers.dicomnode_test_case import DicomnodeTestCase

from dicomnode.dicom.series import DicomSeries
from dicomnode.server.factories.association_events import ReleasedEvent, AssociationTypes
from dicomnode.server.grinders import ListGrinder
from dicomnode.server.input import AbstractInput
from dicomnode.server.nodes import AbstractQueuedPipeline
from dicomnode.server.output import NoOutput
from dicomnode.server.pipeline_tree import InputContainer

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

series_1.set_shared_tag(0x0011_0100, DataElement(0x0011_0100, 'IS', [-1, 2, 0, 1]))
series_2.set_shared_tag(0x0011_0100, DataElement(0x0011_0100, 'IS', [-1, 2, 0, 2]))
series_3.set_shared_tag(0x0011_0100, DataElement(0x0011_0100, 'IS', [-1, 2, 0, 3]))

test_array = []


class TestPipeline(AbstractQueuedPipeline):
  input = {
    "dicoms" : TestInput
  }

  def process(self, input_container: InputContainer):
    datasets: List[Dataset] = input_container["dicoms"]

    pivot = datasets[0]

    test_array.append(pivot.PatientID)

    return NoOutput()


class QueuedPipelineTestCase(DicomnodeTestCase):
  def setUp(self) -> None:
    self.node = TestPipeline()

  def tearDown(self) -> None:
    self.node.close()

  def test_real_dumb(self):
    self.assertEqual(inspect.getsource(self.node.handle_association_released),
                     inspect.getsource(self.node.evt_handlers[evt.EVT_RELEASED]))

  def test_queue(self):
    thread_id_1 = 390
    thread_id_2 = 390
    thread_id_3 = 390

    self.node._updated_patients[thread_id_1] = set([PATIENT_ID_1])
    self.node._updated_patients[thread_id_2] = set([PATIENT_ID_2])
    self.node._updated_patients[thread_id_3] = set([PATIENT_ID_3])
    self.node._patient_locks[PATIENT_ID_1] = (set([thread_id_1]), threading.Lock())
    self.node._patient_locks[PATIENT_ID_2] = (set([thread_id_2]), threading.Lock())
    self.node._patient_locks[PATIENT_ID_3] = (set([thread_id_3]), threading.Lock())

    self.node.data_state.add_images(series_1)
    self.node.data_state.add_images(series_2)
    self.node.data_state.add_images(series_3)

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

    event_1 = evt.Event(AssociationDummy(thread_id_1), evt.EVT_RELEASED) # type: ignore
    event_2 = evt.Event(AssociationDummy(thread_id_2), evt.EVT_RELEASED) # type: ignore
    event_3 = evt.Event(AssociationDummy(thread_id_3), evt.EVT_RELEASED) # type: ignore

    self.node.handle_association_released(event_1)
    self.node.handle_association_released(event_2)
    self.node.handle_association_released(event_3)

    self.node.process_queue.join()