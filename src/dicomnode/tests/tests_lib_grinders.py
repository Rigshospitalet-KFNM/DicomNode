from unittest import TestCase, skipIf

from pydicom import Dataset
from pydicom.uid import SecondaryCaptureImageStorage

from dicomnode.lib.dicom import gen_uid, make_meta
from dicomnode.lib.grinders import identity_grinder, list_grinder

try:
  import numpy

  from dicomnode.lib.grinders import numpy_grinder
  from dicomnode.tests.helpers import generate_numpy_datasets
  NUMPY_IMPORT = True
except ImportError:
  NUMPY_IMPORT = False

def get_test_dataset() -> Dataset:
  dataset = Dataset()
  dataset.SOPClassUID = SecondaryCaptureImageStorage
  dataset.is_little_endian = True
  dataset.is_implicit_VR = True

  return dataset


class GrinderTests(TestCase):
  """
  Grinders are unrelated to any dating websites.
  """

  def setUp(self):
    self.iter = []
    self.patientID_1 = "testID_1"
    self.patientID_2 = "testID_2"

    self.patientName_1 = "TestP1"
    self.patientName_2 = "TestP2"

    self.studyUID_1 = gen_uid()
    self.studyUID_2 = gen_uid()
    self.studyUID_3 = gen_uid()
    self.studyUID_4 = gen_uid()

    self.seriesUID_1 = gen_uid()
    self.seriesUID_2 = gen_uid()
    self.seriesUID_3 = gen_uid()
    self.seriesUID_4 = gen_uid()
    self.seriesUID_5 = gen_uid()


    self.dataset_1 = get_test_dataset()
    self.dataset_1_SOPInstanceUID = gen_uid()
    self.dataset_1.SOPInstanceUID = self.dataset_1_SOPInstanceUID
    self.dataset_1.SeriesInstanceUID = self.seriesUID_1
    self.dataset_1.StudyInstanceUID = self.studyUID_1
    self.dataset_1.PatientID = self.patientID_1
    self.dataset_1.PatientName = self.patientName_1
    self.dataset_1.fix_meta_info()


    self.dataset_2 = get_test_dataset()
    self.dataset_2_SOPInstanceUID = gen_uid()
    self.dataset_2.SOPInstanceUID = self.dataset_2_SOPInstanceUID
    self.dataset_2.SeriesInstanceUID = self.seriesUID_1
    self.dataset_2.StudyInstanceUID = self.studyUID_1
    self.dataset_2.PatientID = self.patientID_1
    self.dataset_2.PatientName = self.patientName_1
    self.dataset_2.fix_meta_info()

    self.dataset_3 = get_test_dataset()
    self.dataset_3_SOPInstanceUID = gen_uid()
    self.dataset_3.SOPInstanceUID = self.dataset_3_SOPInstanceUID
    self.dataset_3.SeriesInstanceUID = self.seriesUID_2
    self.dataset_3.SeriesDescription = "Test Series Description"
    self.dataset_3.StudyInstanceUID = self.studyUID_1
    self.dataset_3.PatientID = self.patientID_1
    self.dataset_3.PatientName = self.patientName_1
    self.dataset_3.fix_meta_info()

    self.dataset_4 = get_test_dataset()
    self.dataset_4_SOPInstanceUID = gen_uid()
    self.dataset_4.SOPInstanceUID = self.dataset_3_SOPInstanceUID
    self.dataset_4.SeriesInstanceUID = self.seriesUID_3
    self.dataset_4.StudyInstanceUID = self.studyUID_2
    self.dataset_4.PatientID = self.patientID_1
    self.dataset_4.PatientName = self.patientName_1
    self.dataset_4.StudyDescription = "Test Study Description"
    self.dataset_4.fix_meta_info()

    self.dataset_5 = get_test_dataset()
    self.dataset_5_SOPInstanceUID = gen_uid()
    self.dataset_5.SOPInstanceUID = self.dataset_4_SOPInstanceUID
    self.dataset_5.SeriesInstanceUID = self.seriesUID_4
    self.dataset_5.StudyInstanceUID = self.studyUID_3
    self.dataset_5.PatientID = self.patientID_2
    self.dataset_5.PatientName = self.patientName_2
    self.dataset_5.fix_meta_info()

    self.dataset_6 = get_test_dataset()
    self.dataset_6_SOPInstanceUID = gen_uid()
    self.dataset_6.SOPInstanceUID = self.dataset_5_SOPInstanceUID
    self.dataset_6.SeriesInstanceUID = self.seriesUID_5
    self.dataset_6.StudyInstanceUID = self.studyUID_4
    self.dataset_6.PatientID = self.patientID_2
    self.dataset_6.PatientName = self.patientName_2
    self.dataset_6.fix_meta_info()

    self.datasets = [self.dataset_1, self.dataset_2, self.dataset_3, self.dataset_4, self.dataset_5, self.dataset_6]

  def test_identity_grinder(self):
    self.assertEqual(id(identity_grinder(self.datasets)), id(self.datasets))

  @skipIf(not NUMPY_IMPORT, "Numpy needed")
  def test_list_grinder(self):
    self.assertNotIsInstance(generate_numpy_datasets(3), list)

    ds_list = list_grinder(generate_numpy_datasets(3))

    self.assertIsInstance(ds_list, list)
    self.assertEqual(len(ds_list), 3)
    for ds in ds_list:
      self.assertIsInstance(ds, Dataset)
