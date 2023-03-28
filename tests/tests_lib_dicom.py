from unittest import TestCase

from pydicom import Dataset

from pydicom.uid import ExplicitVRBigEndian, ExplicitVRLittleEndian, ImplicitVRLittleEndian, CTImageStorage
from dicomnode.lib.dicom import get_tag, make_meta, gen_uid, extrapolate_image_position_patient, extrapolate_image_position_patient_dataset
from dicomnode.lib.exceptions import InvalidDataset

class DicomTestCase(TestCase):
  def test_getTag_success(self):
    ds = Dataset()
    pid = "TestID"
    ds.PatientID = pid # PatientID Tag
    getTag_PatientID = get_tag(0x00100020)
    getTag_PatientName = get_tag(0x00100010)
    self.assertEqual(getTag_PatientID(ds).value, pid)
    self.assertIsNone(getTag_PatientName(ds))

  def test_make_meta_missing_SOPClassUID(self):
    ds = Dataset()
    self.assertRaises(InvalidDataset, make_meta, ds)

  def test_make_meta_ImplicitVRLittleEndian(self):
    ds = Dataset()
    ds.SOPClassUID = CTImageStorage
    ds.is_implicit_VR = True
    ds.is_little_endian = True
    make_meta(ds)
    self.assertEqual(ds.file_meta.TransferSyntaxUID, ImplicitVRLittleEndian)

  def test_make_meta_ExplicitVRLittleEndian(self):
    ds = Dataset()
    ds.SOPClassUID = CTImageStorage
    ds.is_implicit_VR = False
    ds.is_little_endian = True
    make_meta(ds)
    self.assertEqual(ds.file_meta.TransferSyntaxUID, ExplicitVRLittleEndian)

  def test_make_meta_ExplicitVRBigEndian(self):
    ds = Dataset()
    ds.SOPClassUID = CTImageStorage
    ds.is_implicit_VR = False
    ds.is_little_endian = False
    make_meta(ds)
    self.assertEqual(ds.file_meta.TransferSyntaxUID, ExplicitVRBigEndian)

  def test_make_meta_ImplicitVRBigEndian(self):
    ds = Dataset()
    ds.SOPClassUID = CTImageStorage
    ds.is_implicit_VR = True
    ds.is_little_endian = False
    self.assertRaises(InvalidDataset, make_meta, ds)

  def test_extrapolate_image_position_patient(self):
    slice_thickness = 1.0
    orientation = 1
    initial_position = (0.0,0.0,0.0)
    image_orientation = (1.0,0.0,0.0,0.0,1.0,0.0) # Standard coordinate system
    image_number = 1
    slices = 10

    positions = extrapolate_image_position_patient(
      slice_thickness,
      orientation,
      initial_position,
      image_orientation,
      image_number,
      slices
    )

    self.assertListEqual(positions, [
      [0.0,0.0,0.0],
      [0.0,0.0,1.0],
      [0.0,0.0,2.0],
      [0.0,0.0,3.0],
      [0.0,0.0,4.0],
      [0.0,0.0,5.0],
      [0.0,0.0,6.0],
      [0.0,0.0,7.0],
      [0.0,0.0,8.0],
      [0.0,0.0,9.0],
    ])

  def test_extrapolate_image_position_patient_extra_thick(self):
    slice_thickness = 2.0
    orientation = 1
    initial_position = (0.0,0.0,0.0)
    image_orientation = (1.0,0.0,0.0,0.0,1.0,0.0) # Standard coordinate system
    image_number = 1
    slices = 10

    positions = extrapolate_image_position_patient(
      slice_thickness,
      orientation,
      initial_position,
      image_orientation,
      image_number,
      slices
    )

    self.assertListEqual(positions, [
      [0.0,0.0,0.0],
      [0.0,0.0,2.0],
      [0.0,0.0,4.0],
      [0.0,0.0,6.0],
      [0.0,0.0,8.0],
      [0.0,0.0,10.0],
      [0.0,0.0,12.0],
      [0.0,0.0,14.0],
      [0.0,0.0,16.0],
      [0.0,0.0,18.0],
    ])

  def test_extrapolate_image_position_patient_reverse_orientation(self):
    slice_thickness = 1.0
    orientation = -1
    initial_position = (0.0,0.0,0.0)
    image_orientation = (1.0,0.0,0.0,0.0,1.0,0.0) # Standard coordinate system
    image_number = 1
    slices = 10

    positions = extrapolate_image_position_patient(
      slice_thickness,
      orientation,
      initial_position,
      image_orientation,
      image_number,
      slices
    )

    self.assertListEqual(positions, [
      [0.0,0.0,0.0],
      [0.0,0.0,-1.0],
      [0.0,0.0,-2.0],
      [0.0,0.0,-3.0],
      [0.0,0.0,-4.0],
      [0.0,0.0,-5.0],
      [0.0,0.0,-6.0],
      [0.0,0.0,-7.0],
      [0.0,0.0,-8.0],
      [0.0,0.0,-9.0],
    ])

  def test_extrapolate_image_position_different_initial_position(self):
    slice_thickness = 1.0
    orientation = 1
    initial_position = (3.0,-5.0,6.0)
    image_orientation = (1.0,0.0,0.0,0.0,1.0,0.0) # Standard coordinate system
    image_number = 1
    slices = 10

    positions = extrapolate_image_position_patient(
      slice_thickness,
      orientation,
      initial_position,
      image_orientation,
      image_number,
      slices
    )

    self.assertListEqual(positions, [
      [3.0, -5.0, 6.0],
      [3.0, -5.0, 7.0],
      [3.0, -5.0, 8.0],
      [3.0, -5.0, 9.0],
      [3.0, -5.0, 10.0],
      [3.0, -5.0, 11.0],
      [3.0, -5.0, 12.0],
      [3.0, -5.0, 13.0],
      [3.0, -5.0, 14.0],
      [3.0, -5.0, 15.0],
    ])

  def test_extrapolate_image_position_different_offset(self):
    slice_thickness = 1.0
    orientation = 1
    initial_position = (0.0,0.0,0.0)
    image_orientation = (1.0,0.0,0.0,0.0,1.0,0.0) # Standard coordinate system
    image_number = 5
    slices = 10

    positions = extrapolate_image_position_patient(
      slice_thickness,
      orientation,
      initial_position,
      image_orientation,
      image_number,
      slices
    )

    self.assertListEqual(positions, [
      [0.0,0.0,-4.0],
      [0.0,0.0,-3.0],
      [0.0,0.0,-2.0],
      [0.0,0.0,-1.0],
      [0.0,0.0,0.0],
      [0.0,0.0,1.0],
      [0.0,0.0,2.0],
      [0.0,0.0,3.0],
      [0.0,0.0,4.0],
      [0.0,0.0,5.0],
    ])

  def test_extrapolate_image_position_different_dataset_feet_first(self):
    dataset = Dataset()
    dataset.SliceThickness = 1.0
    dataset.PatientPosition = "FF"
    dataset.ImagePositionPatient = [0.0,0.0,0.0]
    dataset.ImageOrientationPatient = [1.0,0.0,0.0,0.0,1.0,0.0] # Standard coordinate system
    dataset.InstanceNumber = 1
    slices = 10

    positions = extrapolate_image_position_patient_dataset(
      dataset,
      slices
    )

    self.assertListEqual(positions, [
      [0.0,0.0,0.0],
      [0.0,0.0,1.0],
      [0.0,0.0,2.0],
      [0.0,0.0,3.0],
      [0.0,0.0,4.0],
      [0.0,0.0,5.0],
      [0.0,0.0,6.0],
      [0.0,0.0,7.0],
      [0.0,0.0,8.0],
      [0.0,0.0,9.0],
    ])

  def test_extrapolate_image_position_different_dataset_Head_first(self):
    dataset = Dataset()
    dataset.SliceThickness = 1.0
    dataset.PatientPosition = "HF"
    dataset.ImagePositionPatient = [0.0,0.0,0.0]
    dataset.ImageOrientationPatient = [1.0,0.0,0.0,0.0,1.0,0.0] # Standard coordinate system
    dataset.InstanceNumber = 1
    slices = 10

    positions = extrapolate_image_position_patient_dataset(
      dataset,
      slices
    )

    self.assertListEqual(positions, [
      [0.0,0.0,0.0],
      [0.0,0.0,-1.0],
      [0.0,0.0,-2.0],
      [0.0,0.0,-3.0],
      [0.0,0.0,-4.0],
      [0.0,0.0,-5.0],
      [0.0,0.0,-6.0],
      [0.0,0.0,-7.0],
      [0.0,0.0,-8.0],
      [0.0,0.0,-9.0],
    ])

  def test_extrapolate_image_position_different_dataset_missing_data(self):
    dataset = Dataset()
    dataset.SliceThickness = 1.0
    dataset.PatientPosition = "HF"
    dataset.ImagePositionPatient = [0.0,0.0,0.0]
    dataset.ImageOrientationPatient = [1.0,0.0,0.0,0.0,1.0,0.0] # Standard coordinate system
    #dataset.InstanceNumber = 1
    slices = 10

    self.assertRaises(InvalidDataset, extrapolate_image_position_patient_dataset, dataset, slices)

  def test_extrapolate_image_position_different_dataset_incorrect_position(self):
    dataset = Dataset()
    dataset.SliceThickness = 1.0
    dataset.PatientPosition = "HF"
    dataset.ImagePositionPatient = [0.0,0.0]
    dataset.ImageOrientationPatient = [1.0,0.0,0.0,0.0,1.0,0.0] # Standard coordinate system
    dataset.InstanceNumber = 1
    slices = 10

    self.assertRaises(InvalidDataset, extrapolate_image_position_patient_dataset, dataset, slices)

  def test_extrapolate_image_position_different_dataset_incorrect_orientation(self):
    dataset = Dataset()
    dataset.SliceThickness = 1.0
    dataset.PatientPosition = "HF"
    dataset.ImagePositionPatient = [0.0,0.0,0.0]
    dataset.ImageOrientationPatient = [1.0,0.0,0.0,1.0,0.0] # Standard coordinate system
    dataset.InstanceNumber = 1
    slices = 10

    self.assertRaises(InvalidDataset, extrapolate_image_position_patient_dataset, dataset, slices)
