"""Tests for src/dicomnode/server/maintenance.py"""

# Python Standard Library
from logging import DEBUG
from datetime import datetime
from unittest import TestCase
from unittest.mock import MagicMock, patch

# Third party modules
from pydicom import Dataset

# Dicomnode modules
from dicomnode.dicom import gen_uid
from dicomnode.server.maintenance import MaintenanceThread
from dicomnode.server.pipeline_tree import PipelineTree
from dicomnode.server.input import AbstractInput

class MaintenanceTestCases(TestCase):
  def test_calculate_time_from_1_before_midnight(self):
    pipeline_tree = PipelineTree(0x00100020, {})
    maintenance_thread = MaintenanceThread(pipeline_tree, 1)

    self.assertEqual(maintenance_thread._seconds_in_a_day,
                     maintenance_thread.calculate_seconds_to_next_maintenance(
                      datetime(2024,6,9,23,59,00)
    ))

  def test_calculate_time_from_midnight(self):
    pipeline_tree = PipelineTree(0x00100020, {})
    maintenance_thread = MaintenanceThread(pipeline_tree, 1)

    self.assertEqual(maintenance_thread._seconds_in_a_day,
                     maintenance_thread.calculate_seconds_to_next_maintenance(
                      datetime(2024,6,9,0,0,0)
    ))

  def test_calculate_time_from_1_after_midnight(self):
    pipeline_tree = PipelineTree(0x00100020, {})
    maintenance_thread = MaintenanceThread(pipeline_tree, 1)

    self.assertEqual(maintenance_thread._seconds_in_a_day - 60,
                     maintenance_thread.calculate_seconds_to_next_maintenance(
                      datetime(2024,6,9,0,1,0)
    ))

  def test_cleaning(self):
    class Input(AbstractInput):
      def validate(self):
        return False

    pipeline_tree = PipelineTree(0x00100020, {
      'input' : Input
    })
    dataset = Dataset()
    dataset.PatientID = "test"
    dataset.SOPInstanceUID = gen_uid()

    pipeline_tree.add_image(dataset)
    maintenance_thread = MaintenanceThread(pipeline_tree, 0)

    with self.assertLogs('dicomnode', DEBUG) as recorded_logs:
      maintenance_thread.maintenance()

    self.assertEqual(pipeline_tree.images, 0)


