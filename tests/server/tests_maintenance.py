"""Tests for src/dicomnode/server/maintenance.py"""

# Python Standard Library
from logging import DEBUG
from datetime import datetime

# Third party modules
from pydicom import Dataset

# Dicomnode modules
from dicomnode.dicom import gen_uid, DicomIdentifier
from dicomnode.server.maintenance import MaintenanceThread
from dicomnode.server.pipeline_storage import ReactivePipelineStorage
from dicomnode.config import config_from_raw
from dicomnode.server.input import AbstractInput

from tests.helpers.dicomnode_test_case import DicomnodeTestCase

class MaintenanceTestCases(DicomnodeTestCase):
  def test_calculate_time_from_1_before_midnight(self):
    storage_to_clean = ReactivePipelineStorage({}, config_from_raw())
    maintenance_thread = MaintenanceThread(storage_to_clean, 1)

    self.assertEqual(maintenance_thread._seconds_in_a_day,
                     maintenance_thread.calculate_seconds_to_next_maintenance(
                      datetime(2024,6,9,23,59,00)
    ))

  def test_calculate_time_from_midnight(self):
    storage_to_clean = ReactivePipelineStorage({}, config_from_raw())
    maintenance_thread = MaintenanceThread(storage_to_clean, 1)

    self.assertEqual(maintenance_thread._seconds_in_a_day,
                     maintenance_thread.calculate_seconds_to_next_maintenance(
                      datetime(2024,6,9,0,0,0)
    ))

  def test_calculate_time_from_1_after_midnight(self):
    storage_to_clean = ReactivePipelineStorage({}, config_from_raw())
    maintenance_thread = MaintenanceThread(storage_to_clean, 1)

    self.assertEqual(maintenance_thread._seconds_in_a_day - 60,
                     maintenance_thread.calculate_seconds_to_next_maintenance(
                      datetime(2024,6,9,0,1,0)
    ))

  def test_cleaning(self):
    class Input(AbstractInput):
      def validate(self):
        return False

    storage_to_clean = ReactivePipelineStorage({"input" : Input}, config_from_raw())
    dataset = Dataset()
    dataset.PatientID = "test"
    dataset.SOPInstanceUID = gen_uid()

    storage_to_clean.add_image(dataset)
    maintenance_thread = MaintenanceThread(storage_to_clean, 0)

    with self.assertLogs('dicomnode', DEBUG) as recorded_logs:
      maintenance_thread.maintenance()
