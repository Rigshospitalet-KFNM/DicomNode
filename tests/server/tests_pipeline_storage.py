# Python standard library
from threading import Thread
from time import sleep
from typing import Dict, List, Tuple

# Third Party modules
from pydicom import Dataset
from pydicom.uid import SecondaryCaptureImageStorage

# Dicomnode modules
from dicomnode.constants import DICOMNODE_LOGGER_NAME
from dicomnode.lib.exceptions import InvalidDataset
from dicomnode.dicom import make_meta, DicomIdentifier, gen_uid
from dicomnode.server.input import AbstractInput
from dicomnode.config import config_from_raw
from dicomnode.server.pipeline_storage import PipelineStorage
from dicomnode.server.patient_node import PatientNode

# Dicomnode Test helper modules
from tests.helpers.dicomnode_test_case import DicomnodeTestCase


def generate_series(patient_index, num_datasets):
  series_uid = gen_uid()

  datasets = []
  for i in range(num_datasets):
    ds = Dataset()
    ds.PatientID = str(patient_index)
    ds.PatientName = f"Patient {patient_index}"
    ds.SOPInstanceUID = gen_uid()
    ds.InstanceNumber = i + 1
    ds.SOPClassUID = SecondaryCaptureImageStorage
    ds.SeriesInstanceUID = series_uid
    #make_meta(ds)

    datasets.append(ds)

  return datasets

class PipelineStorageTestCase(DicomnodeTestCase):
  def test_pipeline_storage_stress_test(self):
    """Create a large amount of threads that will behave as if they where
    associations, then those thread spam an pipeline storage input with 3 series

    These thread interleave patients so:
    Thread 1 sends Patient 1,2,3 and Thread 2 send 2,3,4 and Thread N sends
    N,1,2
    """

    # Test Constants
    num_stressors = 20 # Also num threads but w/e
    datasets_per_series = 50
    series_per_patient = 3


    class HungryInput(AbstractInput):
      required_tags = ["SOPInstanceUID", 0x0020_000E] # Series InstanceUID

      enforce_single_series = False

      def validate(self) -> bool:
        series_UIDs = set()

        for dataset in self:
          series_UIDs.add(dataset.SeriesInstanceUID)

        num_series_UIDs = len(series_UIDs)
        valid = num_series_UIDs == series_per_patient

        return valid


    target = PipelineStorage(
      {"input" : HungryInput},
      config_from_raw()
    )

    thread_datasets = []
    successful_extractions: Dict[int, List[Tuple[str,PatientNode]]] = {}
    failed_datasets = {}

    for patient_index in range(num_stressors): #[1,2,3...]
      # Not sure if this is one of those fuck you it's way to complicated.
      # Any who It just generates N series per patient

      thread_datasets.append([
        generate_series(((patient_index + series_num) % num_stressors) + 1 , datasets_per_series)
        for series_num in range(series_per_patient)
      ])

    def thread_target_function(thread_index):
      series = thread_datasets[thread_index]

      exported_series = 0

      for a_series in series:
        for dataset in a_series:
          target.add_image(dataset)
        exported_series += 1
        #print(f"Thread {thread_index + 1} completed lap: {exported_series} / {series_per_patient}")
        sleep(0.00001)

      stuff, failed = target.extract_input_container()

      successful_extractions[thread_index] = stuff
      failed_datasets[thread_index] = failed

    threads = [
      Thread(target=thread_target_function, args=(thread_index,),name=f"Pipeline Storage {thread_index + 1} Test Thread")
      for thread_index in range(num_stressors )
    ]
    with self.assertLogs(DICOMNODE_LOGGER_NAME):
      for thread in reversed(threads):
        thread.start()

      for thread in threads:
        thread.join()

      self.assertEqual(0, len(target.storage))
      self.assertEqual(0, len(target.thread_registration))
      self.assertEqual(0, len(target.thread_additions))
      self.assertEqual(0, len(target.failed_additions))

      for failed_to_add in failed_datasets.values():
        self.assertEqual(len(failed_to_add), 0)

    for patient_nodes in successful_extractions.values():
      for patient_id, node in patient_nodes:
        self.assertEqual(len(node), series_per_patient * datasets_per_series)

  def test_failing_to_add_accumulates_datasets(self):
    class DenyingInput(AbstractInput):
      def validate(self):
        return False

      def add_image(self, dicom: Dataset) -> int:
        raise InvalidDataset

    target = PipelineStorage({
      'Anger' : DenyingInput
    }, config=config_from_raw())

    target.add_images(generate_series(1, 5))

  def test_can_string_convert(self):
    class AcceptingInput(AbstractInput):
      def validate(self):
        return False

      def add_image(self, dicom: Dataset) -> int:
        self.storage.store_image(dicom)
        return 1

    target = PipelineStorage({
      'Anger' : AcceptingInput
    }, config=config_from_raw())

    target.add_images(generate_series(1, 5))
    self.assertIsInstance(str(target), str)
