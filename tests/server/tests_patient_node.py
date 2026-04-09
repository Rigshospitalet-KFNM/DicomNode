# Python Standard Library

# Third party modules
from pydicom import Dataset
from pydicom.uid import PositronEmissionTomographyImageStorage, CTImageStorage,\
  SecondaryCaptureImageStorage

# Dicomnode modules
from dicomnode.dicom import gen_uid
from dicomnode.lib.exceptions import InvalidDataset
from dicomnode.config import config_from_raw
from dicomnode.server.patient_node import PatientNode
from dicomnode.server.input import AbstractInput

# Dicomnode Test modules
from tests.helpers.dicomnode_test_case import DicomnodeTestCase

def generate_dataset(study_date, class_ = SecondaryCaptureImageStorage):
  ds = Dataset()
  ds.SOPInstanceUID = gen_uid()
  ds.SOPClassUID = class_
  ds.StudyDate = study_date
  return ds

class Input(AbstractInput):
  required_tags = ['SOPInstanceUID', 'StudyDate']

  enforce_single_study_date = True

  def validate(self) -> bool:
    return True

class PETInput(AbstractInput):
  required_tags = [
    "StudyDate",
    "SOPInstanceUID"
  ]

  enforce_single_study_date = True

  required_values = {
    "SOPClassUID" : PositronEmissionTomographyImageStorage
  }

  def validate(self) -> bool:
    return 0 < self.images

class CTInput(AbstractInput):
  required_tags = [
    "StudyDate",
    "SOPInstanceUID"
  ]

  enforce_single_study_date = True

  required_values = {
    "SOPClassUID" : CTImageStorage
  }

  def validate(self) -> bool:
    return 0 < self.images

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


  def test_patient_node_raises_on_no_accept(self):
    node = PatientNode({ 'node' : Input}, config_from_raw())

    self.assertRaises(InvalidDataset, node.add_dataset, Dataset())

  def test_input_study_sets_both_inputs_study_date(self):
    node = PatientNode({
      'ct' : CTInput,
      'pet' : PETInput
    }, config_from_raw())

    past_study_date = "20190101"
    study_date = "20200101"
    future_study_date = "20210101"

    node.add_dataset(generate_dataset(study_date, PositronEmissionTomographyImageStorage))

    for input_ in node:
      self.assertEqual(input_.study_date, study_date)

    self.assertRaises(InvalidDataset, node.add_dataset, generate_dataset(past_study_date, PositronEmissionTomographyImageStorage))
    self.assertRaises(InvalidDataset, node.add_dataset, generate_dataset(future_study_date, PositronEmissionTomographyImageStorage))
    self.assertRaises(InvalidDataset, node.add_dataset, generate_dataset(past_study_date, CTImageStorage))
    self.assertRaises(InvalidDataset, node.add_dataset, generate_dataset(future_study_date, CTImageStorage))

    node.add_dataset(generate_dataset(study_date, CTImageStorage))
    self.assertTrue(node.validate())
