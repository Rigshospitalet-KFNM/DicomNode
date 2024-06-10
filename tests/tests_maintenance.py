"""Tests for src/dicomnode/server/maintenance.py"""

# Python Standard Library
from datetime import datetime
from unittest import TestCase

# Third party modules

# Dicomnode modules
from dicomnode.server.maintenance import MaintenanceThread
from dicomnode.server.pipeline_tree import PipelineTree

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

