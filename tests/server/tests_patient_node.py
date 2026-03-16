# Python Standard Library

# Third party modules
from pydicom import Dataset

# Dicomnode modules
from dicomnode.dicom import gen_uid
from dicomnode.lib.exceptions import InvalidDataset
from dicomnode.server.dicomnode_config import config_from_raw
from dicomnode.server.patient_node import PatientNode
from dicomnode.server.input import AbstractInput

# Dicomnode Test modules
from tests.helpers.dicomnode_test_case import DicomnodeTestCase

def generate_dataset(study_date):
  ds = Dataset()
  ds.SOPInstanceUID = gen_uid()
  ds.StudyDate = study_date
  return ds

class Input(AbstractInput):
  required_tags = ['SOPInstanceUID', 'StudyDate']

  enforce_single_study_date = True

  def validate(self) -> bool:
    return True

class PatientNodeTestCase(DicomnodeTestCase):
  def test_patient_node_sets_study_date(self):
    config = config_from_raw()

    study_date = "20200101"
    not_study_date = "20210101"

    node = PatientNode({ 'node' : Input}, config)
    node.add_dataset(generate_dataset(study_date))

    self.assertRaises(InvalidDataset, node.add_dataset, generate_dataset(not_study_date))

    self.assertEqual(node.study_date, study_date)
    for input_ in node:
      self.assertEqual(input_.study_date, study_date)
