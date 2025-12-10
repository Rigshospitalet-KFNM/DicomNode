
from pathlib import Path
import shutil

from pydicom import Dataset
from pydicom.uid import UID, MediaStorageDirectoryStorage, SecondaryCaptureImageStorage
from unittest import TestCase, skip

from tests.helpers import generate_numpy_datasets, bench
from tests.helpers.dicomnode_test_case import DicomnodeTestCase

from dicomnode.dicom import gen_uid, make_meta
from dicomnode.lib.io import load_dicom
from dicomnode.data_structures.image_tree import DicomTree, SeriesTree, StudyTree, PatientTree, IdentityMapping, ImageTreeInterface

def get_test_dataset() -> Dataset:
  dataset = Dataset()
  dataset.SOPClassUID = SecondaryCaptureImageStorage

  return dataset

class lib_imageTree(DicomnodeTestCase):
  def setUp(self) -> None:
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



    self.dataset_2 = get_test_dataset()
    self.dataset_2_SOPInstanceUID = gen_uid()
    self.dataset_2.SOPInstanceUID = self.dataset_2_SOPInstanceUID
    self.dataset_2.SeriesInstanceUID = self.seriesUID_1
    self.dataset_2.StudyInstanceUID = self.studyUID_1
    self.dataset_2.PatientID = self.patientID_1
    self.dataset_2.PatientName = self.patientName_1


    self.dataset_3 = get_test_dataset()
    self.dataset_3_SOPInstanceUID = gen_uid()
    self.dataset_3.SOPInstanceUID = self.dataset_3_SOPInstanceUID
    self.dataset_3.SeriesInstanceUID = self.seriesUID_2
    self.dataset_3.SeriesDescription = "Test Series Description"
    self.dataset_3.StudyInstanceUID = self.studyUID_1
    self.dataset_3.PatientID = self.patientID_1
    self.dataset_3.PatientName = self.patientName_1


    self.dataset_4 = get_test_dataset()
    self.dataset_4_SOPInstanceUID = gen_uid()
    self.dataset_4.SOPInstanceUID = self.dataset_3_SOPInstanceUID
    self.dataset_4.SeriesInstanceUID = self.seriesUID_3
    self.dataset_4.StudyInstanceUID = self.studyUID_2
    self.dataset_4.PatientID = self.patientID_1
    self.dataset_4.PatientName = self.patientName_1
    self.dataset_4.StudyDescription = "Test Study Description"


    self.dataset_5 = get_test_dataset()
    self.dataset_5_SOPInstanceUID = gen_uid()
    self.dataset_5.SOPInstanceUID = self.dataset_4_SOPInstanceUID
    self.dataset_5.SeriesInstanceUID = self.seriesUID_4
    self.dataset_5.StudyInstanceUID = self.studyUID_3
    self.dataset_5.PatientID = self.patientID_2
    self.dataset_5.PatientName = self.patientName_2

    self.dataset_6 = get_test_dataset()
    self.dataset_6_SOPInstanceUID = gen_uid()
    self.dataset_6.SOPInstanceUID = self.dataset_5_SOPInstanceUID
    self.dataset_6.SeriesInstanceUID = self.seriesUID_5
    self.dataset_6.StudyInstanceUID = self.studyUID_4
    self.dataset_6.PatientID = self.patientID_2
    self.dataset_6.PatientName = self.patientName_2

    self.datasets = [self.dataset_1, self.dataset_2, self.dataset_3, self.dataset_4, self.dataset_5, self.dataset_6]
    [make_meta(ds) for ds in self.datasets]
    self.SOPInstanceUIDs = [
      self.dataset_1_SOPInstanceUID,
      self.dataset_2_SOPInstanceUID,
      self.dataset_3_SOPInstanceUID,
      self.dataset_4_SOPInstanceUID,
      self.dataset_5_SOPInstanceUID,
      self.dataset_6_SOPInstanceUID]


    self.empty_dataset = Dataset()


  ##### Test for creating the data structure DicomTrees #####
  def test_create_DicomTree_createEmptyDataset(self):
    dt = DicomTree()
    # Assertions
    self.assertEqual(dt.images, 0)
    self.assertEqual(len(dt.data), 0)
    self.assertTrue(isinstance(dt.data, dict))

  def test_create_DicomTree_OnePicture(self):
    dt = DicomTree(self.dataset_1)
    pt = dt.data[self.dataset_1.PatientID]
    stt = pt.data[self.dataset_1.StudyInstanceUID]
    seTree = stt.data[self.dataset_1.SeriesInstanceUID]
    # Assertions
    self.assertEqual(dt.images, 1)
    self.assertEqual(len(dt.data), 1)
    self.assertTrue(isinstance(pt, PatientTree))
    self.assertEqual(pt.images, 1)
    self.assertTrue(isinstance(stt, StudyTree))
    self.assertEqual(stt.images, 1)
    self.assertTrue(isinstance(seTree, SeriesTree))
    self.assertEqual(seTree.images, 1)
    self.assertEqual(seTree.data[self.dataset_1.SOPInstanceUID], self.dataset_1)

  def test_create_DicomTree_ManyPictures(self):
    dt = DicomTree(self.datasets)
    # Assertions
    self.assertEqual(dt.images, len(self.datasets))

  def test_DicomTree_add_images(self):
    dt = DicomTree()
    dt.add_images(self.datasets)
    # Assertions
    self.assertEqual(dt.images, len(self.datasets))

  def test_DicomTree_add_image(self):
    dt = DicomTree()
    dt.add_image(self.dataset_1)
    pt = dt.data[self.dataset_1.PatientID]
    stt = pt.data[self.dataset_1.StudyInstanceUID]
    seTree = stt.data[self.dataset_1.SeriesInstanceUID]
    # Assertions
    self.assertEqual(dt.images, 1)
    self.assertEqual(dt.images, 1)
    self.assertEqual(len(dt.data), 1)
    self.assertTrue(isinstance(pt, PatientTree))
    self.assertEqual(pt.images, 1)
    self.assertTrue(isinstance(stt, StudyTree))
    self.assertEqual(stt.images, 1)
    self.assertTrue(isinstance(seTree, SeriesTree))
    self.assertEqual(seTree.images, 1)
    self.assertEqual(seTree.data[self.dataset_1.SOPInstanceUID], self.dataset_1)

  def test_DicomTree_to_str_method_pictures(self):
    dt = DicomTree(self.datasets)
    self.maxDiff = 10000
    self.assertEqual(str(dt), f"""Dicom Tree with 6 images
  Patient StudyTree of TestP1 with 4 images
    Undefined Study Description with 3 images with Series:
      Tree of Undefined Series with 2 images
      Tree of Test Series Description with 1 images
    Tree of Test Study Description with 1 images with Series:
      Tree of Undefined Series with 1 images
  Patient StudyTree of TestP2 with 2 images
    Undefined Study Description with 1 images with Series:
      Tree of Undefined Series with 1 images
    Undefined Study Description with 1 images with Series:
      Tree of Undefined Series with 1 images
""")
  # Invalid Adding Data to trees
  def test_DicomTree_add_Invalid_dataset(self):
    dt = DicomTree()
    self.assertRaises(ValueError, dt.add_image, self.empty_dataset)

  def test_PatientTree_add_Empty_dataset(self):
    st = PatientTree()
    self.assertRaises(ValueError, st.add_image, self.empty_dataset)

  def test_PatientTree_add_missing_patient_ID(self):
    new_DS = Dataset()
    new_DS.StudyInstanceUID = self.studyUID_1

    pt = PatientTree()
    self.assertRaises(ValueError, pt.add_image, new_DS)

  def test_StudyTree_add_Empty_dataset(self):
    st = StudyTree()
    self.assertRaises(ValueError, st.add_image, self.empty_dataset)

  def test_StudyTree_add_missing_patient_ID(self):
    new_DS = Dataset()
    new_DS.StudyInstanceUID = self.studyUID_1
    st = StudyTree()
    self.assertRaises(ValueError, st.add_image, new_DS)


  def test_SeriesTree_add_Empty_dataset(self):
    st = SeriesTree()
    self.assertRaises(ValueError, st.add_image, self.empty_dataset)

  def test_multiple_patient_in_Patient_tree(self):
    PT = PatientTree(self.dataset_1)
    self.assertRaises(KeyError, PT.add_image, self.dataset_5)

  def test_multiple_Study_in_StudyTree(self):
    ST = StudyTree(self.dataset_1)
    self.assertRaises(KeyError, ST.add_image, self.dataset_4)

  def test_multiple_Series_in_SeriesTree(self):
    SeT = SeriesTree(self.dataset_1)
    self.assertRaises(KeyError, SeT.add_image, self.dataset_3)

  def test_duplicate_image_addition(self):
    DT = DicomTree(self.dataset_1)
    self.assertRaises(ValueError,DT.add_image,self.dataset_1)

  def test_no_SOPInstanceUID(self):
    SeT = SeriesTree()
    ds = Dataset()
    ds.SeriesInstanceUID = self.seriesUID_1
    self.assertRaises(ValueError, SeT.add_image, ds)

  def test_discover(self):
    dicom_path = Path(self._testMethodName)
    DT = DicomTree(self.datasets)
    DT.save_tree(dicom_path)
    dummyFile =  dicom_path / "dummy.txt"
    dummyFile.touch()

    new_DT = DicomTree()
    new_DT.discover(dicom_path)

    self.assertEqual(new_DT.images, len(self.datasets))

    for ds in new_DT:
      self.assertIn(ds, self.datasets)

    self.assertTrue(dicom_path.exists())
    self.assertTrue(dicom_path.is_dir())

    shutil.rmtree(dicom_path)


  # Functions of DicomTree
  def test_DicomTree_apply_mapping(self):
    def writeModality(ds : Dataset):
      ds.Modality = 'OT'

    dt = DicomTree(self.datasets)
    dt.map(writeModality)

    for ds in self.datasets:
      self.assertEqual(ds.Modality, "OT")

  def test_maps_with_returnVals(self):
    DT = DicomTree(self.datasets)
    def getSOP_UID(dataset: Dataset)-> UID:
      return dataset.SOPInstanceUID
    SOPs = DT.map(getSOP_UID)

    for SOP in SOPs:
      self.assertIn(SOP, self.SOPInstanceUIDs)


  def test_DicomTree_apply_mapping_with_empty_IM(self):
    def writeModality(ds : Dataset):
      ds.Modality = 'OT'

    IM = IdentityMapping()
    dt = DicomTree(self.datasets)
    dt.map(writeModality, IM)

    for ds in self.datasets:
      self.assertEqual(ds.Modality, "OT")

  def test_DicomTree_trimTree(self):
    def filterFunc(ds):
      return ds.PatientID == self.patientID_1

    dt = DicomTree(self.datasets)
    removed_images = dt.trim_tree(filterFunc)
    self.assertEqual(removed_images + dt.images, len(self.datasets))

  # Interface is actually an interface
  def test_interface(self):
    class Tree(ImageTreeInterface):
      pass

    self.assertRaises(TypeError, Tree)

  ### Attribute Tests ###

  def test_setitem(self):
    ST = SeriesTree([self.dataset_1])
    self.assertRaises(TypeError, ST.__setitem__, 1, self.dataset_2)
    self.assertRaises(TypeError, ST.__setitem__, "asd", 1)
    ST[self.dataset_2_SOPInstanceUID] = self.dataset_2

  def test_get_item(self):
    ST = SeriesTree([self.dataset_1])
    self.assertEqual(ST[self.dataset_1_SOPInstanceUID], self.dataset_1)

  def test_delitem(self):
    ST = SeriesTree([self.dataset_1])
    del ST[self.dataset_1_SOPInstanceUID]
    self.assertEqual(ST.images, 0)
    self.assertNotIn(self.dataset_1_SOPInstanceUID, ST)
    StudyT = StudyTree([self.dataset_1, self.dataset_2])
    del StudyT[self.dataset_1.SeriesInstanceUID]
    self.assertEqual(StudyT.images, 0)


###### IdentityMapping #########
  def test_create_IdentityMapping(self):
    im = IdentityMapping()

    self.assertTrue(isinstance(im.StudyUIDMapping, dict))
    self.assertTrue(isinstance(im.SeriesUIDMapping, dict))
    self.assertTrue(isinstance(im.SOP_UIDMapping, dict))
    self.assertTrue(isinstance(im.PatientMapping, dict))

    self.assertEqual(len(im.StudyUIDMapping), 0)
    self.assertEqual(len(im.SeriesUIDMapping), 0)
    self.assertEqual(len(im.SOP_UIDMapping), 0)
    self.assertEqual(len(im.PatientMapping), 0)
    self.assertEqual(im.prefix_size, 4)

  def test_create_IM_with_DT(self):
    im = IdentityMapping()
    dt = DicomTree(self.datasets)
    im.fill_from_DicomTree(dt)

  def test_IdentityMapping_add_StudyUID(self):
    im = IdentityMapping()
    uid = im.add_StudyUID(self.studyUID_1)
    uid2 = im.StudyUIDMapping[self.studyUID_1.name]
    self.assertEqual(uid, uid2)

  def test_IdentityMapping_add_SeriesUID(self):
    im = IdentityMapping()
    uid = im.add_SeriesUID(self.seriesUID_1)
    uid2 = im.SeriesUIDMapping[self.seriesUID_1.name]
    self.assertEqual(uid, uid2)

  def test_IdentityMapping_add_SOP_UID(self):
    im = IdentityMapping()
    uid =  im.add_SOP_UID(self.dataset_1_SOPInstanceUID)
    uid2 = im.SOP_UIDMapping[self.dataset_1_SOPInstanceUID.name]
    self.assertEqual(uid, uid2)

  def test_IdentityMapping_get_UID(self):
    im = IdentityMapping()
    uid = im.add_SOP_UID(self.dataset_1_SOPInstanceUID)
    self.assertEqual(im[self.dataset_1_SOPInstanceUID], uid)

  def test_IdentityMapping_get_non_existence(self):
    im = IdentityMapping()
    self.assertRaises(KeyError, im.__getitem__, self.dataset_1_SOPInstanceUID)


  def test_Getting_PatientID(self):
    im = IdentityMapping()
    ret_1 = im.add_Patient("Patient")
    ret_2 = im.add_Patient("Patient")
    self.assertEqual(ret_1, ret_2)

  def test_IdentityMapping_get_mapping(self):
    prefix = "pp_"
    im = IdentityMapping()
    dt = DicomTree(self.datasets)
    im.fill_from_DicomTree(dt, prefix)

    sop_uid_ret_1 = im.get_mapping(self.dataset_1_SOPInstanceUID)
    sop_str_ret_1 = im.get_mapping(self.dataset_1_SOPInstanceUID.name)

    study_uid_ret_1 = im.get_mapping(self.studyUID_1)
    study_str_ret_1 = im.get_mapping(self.studyUID_1.name)

    series_uid_ret_1 = im.get_mapping(self.seriesUID_1)
    series_str_ret_1 = im.get_mapping(self.seriesUID_1.name)

    patientID = im.get_mapping(self.patientID_1)

    self.assertIsNotNone(patientID)
    self.assertEqual(patientID[:len(prefix)], prefix) # type: ignore

    self.assertEqual(sop_uid_ret_1, sop_str_ret_1)
    self.assertEqual(study_uid_ret_1, study_str_ret_1)
    self.assertEqual(series_uid_ret_1, series_str_ret_1)

    self.assertIsNone(im.get_mapping(gen_uid()))

  def test_IdentityMapping_str(self):
    im = IdentityMapping()
    dt = DicomTree(self.datasets)
    im.fill_from_DicomTree(dt)

    self.assertEqual(str(im), """Identity Mapping
  Patient Mapping
{'testID_1': 'AnonymizedPatientID_0', 'testID_2': 'AnonymizedPatientID_1'}
  Study Mapping with 4 Mappings
  Series Mapping with 5
  SOP Mapping with 5 Mappings""")

  # Saving tree
  def test_save_tree(self):
    testDir = Path(self._testMethodName)
    DT = DicomTree(self.datasets)
    DT.save_tree(testDir)

    self.assertTrue((testDir / self.patientID_1).exists())
    self.assertTrue((testDir / self.patientID_2).exists())

    shutil.rmtree(testDir)

  def test_save_file(self):
    dicom_path = Path(self._testMethodName)
    DT = DicomTree(self.dataset_1)
    DT.save_tree(dicom_path)
    self.assertTrue(dicom_path.exists())
    self.assertTrue(dicom_path.is_file())

    dicom = load_dicom(dicom_path)

    self.assertEqual(dicom, self.dataset_1)
    dicom_path.unlink()

  def test_save_to_dir(self):
    dir_path = Path(self._testMethodName)
    dir_path.mkdir()
    DT = DicomTree(self.dataset_1)
    DT.save_tree(dir_path)

    dicom_path = dir_path / (self.dataset_1_SOPInstanceUID.name + '.dcm')
    self.assertTrue(dicom_path.exists())
    self.assertTrue(dicom_path.is_file())

    dicom = load_dicom(dicom_path)

    self.assertEqual(dicom, self.dataset_1)
    shutil.rmtree(dir_path)


  # Test Iteration
  def test_iteration(self):
    DT = DicomTree(self.datasets)
    for image in DT:
      self.assertIn(image, self.datasets)

    for series in DT.series():
      self.assertIsInstance(series, SeriesTree)

    for series in DT.studies():
      self.assertIsInstance(series, StudyTree)

    for PT in DT.patients():
      self.assertIsInstance(PT, PatientTree)


  def test_image_tree_contains(self):
    series_tree = SeriesTree([self.dataset_1])
    self.assertIn(self.dataset_1[0x00080018], series_tree)

    # You might argue that should raise an error here
    # You might also argue that the functionality should be very very different.
    self.assertNotIn(123, series_tree)


  @bench
  def performance_max_recursion(self):
    DT = DicomTree()
    datasets = set()
    studies = 2000

    for i in range(studies):
      for ds in generate_numpy_datasets(1, Cols=2,Rows=2, PatientID=f"patient_{i}"):
        datasets.add(ds.SOPInstanceUID.name)
        DT.add_image(ds)

    self.assertEqual(DT.images,studies)
    for ds in DT:
      self.assertIn(ds.SOPInstanceUID.name, datasets)