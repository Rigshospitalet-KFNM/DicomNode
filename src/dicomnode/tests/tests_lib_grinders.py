from unittest import TestCase, skipIf

from pydicom import Dataset
from pydicom.uid import SecondaryCaptureImageStorage

from dicomnode.lib.exceptions import InvalidDataset
from dicomnode.lib.imageTree import DicomTree
from dicomnode.lib.dicom import gen_uid, make_meta
from dicomnode.lib.grinders import identity_grinder, list_grinder, many_meta_grinder, dicom_tree_grinder

import numpy
import logging
from dicomnode.lib.grinders import numpy_grinder
from dicomnode.tests.helpers import generate_numpy_datasets

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

  def test_list_grinder(self):
    self.assertNotIsInstance(generate_numpy_datasets(3), list)

    ds_list = list_grinder(generate_numpy_datasets(3))

    self.assertIsInstance(ds_list, list)
    self.assertEqual(len(ds_list), 3)
    for ds in ds_list:
      self.assertIsInstance(ds, Dataset)


  def test_meta_grinder(self):
    meta_grinder = many_meta_grinder(list_grinder, dicom_tree_grinder, identity_grinder)
    ds_list, dicom_tree, identity = meta_grinder(self.datasets)

    self.assertListEqual(ds_list, self.datasets)
    self.assertIsNot(ds_list, self.datasets)

    self.assertIsInstance(dicom_tree, DicomTree)
    self.assertIs(identity, self.datasets)


  def test_numpy_grinder_rescale(self):
    images = 10
    rows = 11
    cols = 12

    cube = numpy_grinder(generate_numpy_datasets(
      images, Cols=cols, Rows=rows
    ))

    self.assertIsInstance(cube, numpy.ndarray)
    self.assertEqual(cube.shape, (images, rows, cols))
    self.assertEqual(cube.dtype, numpy.float64)

  def test_numpy_grinder_uint16(self):
    images = 10
    rows = 11
    cols = 12

    cube = numpy_grinder(generate_numpy_datasets(
      images, Cols=cols, Rows=rows, rescale=False
    ))

    self.assertIsInstance(cube, numpy.ndarray)
    self.assertEqual(cube.shape, (images,rows, cols))
    self.assertEqual(cube.dtype, numpy.uint16)

  def test_numpy_grinder_int16(self):
    images = 10
    rows = 11
    cols = 12

    cube = numpy_grinder(generate_numpy_datasets(
      images, Cols=cols, Rows=rows, rescale=False, PixelRepresentation=1
    ))

    self.assertIsInstance(cube, numpy.ndarray)
    self.assertEqual(cube.shape, (images,rows, cols))
    self.assertEqual(cube.dtype, numpy.int16)

  def test_numpy_sorting(self):
    ds = list(generate_numpy_datasets(3, Rows=2, Cols=2, rescale=False))

    ds[0].PixelData = numpy.array([[1,1],[2,2]], dtype=numpy.uint16).tobytes()
    ds[1].PixelData = numpy.array([[5,5],[6,6]], dtype=numpy.uint16).tobytes()
    ds[2].PixelData = numpy.array([[3,3],[4,4]], dtype=numpy.uint16).tobytes()

    ds[1].InstanceNumber = 3
    ds[2].InstanceNumber = 2

    cube = numpy_grinder(ds)

    self.assertTrue((cube == numpy.array([[[1,1],[2,2]],[[3,3],[4,4]],[[5,5],[6,6]]])).all())

  def test_numpy_no_sorting(self):
    ds = list(generate_numpy_datasets(3, Rows=2, Cols=2, rescale=False))

    ds[0].PixelData = numpy.array([[1,1],[2,2]], dtype=numpy.uint16).tobytes()
    ds[1].PixelData = numpy.array([[5,5],[6,6]], dtype=numpy.uint16).tobytes()
    ds[2].PixelData = numpy.array([[3,3],[4,4]], dtype=numpy.uint16).tobytes()

    del ds[0].InstanceNumber
    del ds[1].InstanceNumber
    del ds[2].InstanceNumber

    cube = numpy_grinder(ds)

    self.assertLogs("Instance Number not present in dataset, arbitrary ordering of datasets", logging.WARNING)
    self.assertTrue((cube == numpy.array([[[1,1],[2,2]],[[5,5],[6,6]],[[3,3],[4,4]]])).all())

  # Numpy Grinder edge cases
  def test_numpy_float(self):
    ds = list(generate_numpy_datasets(3, Rows=2, Cols=2, rescale=False, Bits=32))

    del ds[0].PixelData
    del ds[1].PixelData
    del ds[2].PixelData

    arr_1 = numpy.random.uniform(0,1, size=(2,2)).astype(dtype=numpy.float32)
    arr_2 = numpy.random.uniform(0,1, size=(2,2)).astype(dtype=numpy.float32)
    arr_3 = numpy.random.uniform(0,1, size=(2,2)).astype(dtype=numpy.float32)


    ds[0].FloatPixelData = arr_1.tobytes()
    ds[1].FloatPixelData = arr_2.tobytes()
    ds[2].FloatPixelData = arr_3.tobytes()

    cube = numpy_grinder(ds)


    self.assertTrue((cube[0,:,:] == arr_1).all())
    self.assertTrue((cube[1,:,:] == arr_2).all())
    self.assertTrue((cube[2,:,:] == arr_3).all())

  def test_numpy_double_float(self):
    ds = list(generate_numpy_datasets(3, Rows=2, Cols=2, rescale=False, Bits=64))

    del ds[0].PixelData
    del ds[1].PixelData
    del ds[2].PixelData

    arr_1 = numpy.random.uniform(0,1, size=(2,2)).astype(dtype=numpy.float64)
    arr_2 = numpy.random.uniform(0,1, size=(2,2)).astype(dtype=numpy.float64)
    arr_3 = numpy.random.uniform(0,1, size=(2,2)).astype(dtype=numpy.float64)

    ds[0].DoubleFloatPixelData = arr_1.tobytes()
    ds[1].DoubleFloatPixelData = arr_2.tobytes()
    ds[2].DoubleFloatPixelData = arr_3.tobytes()

    cube = numpy_grinder(ds)

    self.assertTrue((cube[0,:,:] == arr_1).all())
    self.assertTrue((cube[1,:,:] == arr_2).all())
    self.assertTrue((cube[2,:,:] == arr_3).all())

  def test_numpy_24bit_pictures(self):
    ds = list(generate_numpy_datasets(3, Rows=2, Cols=2, rescale=False, Bits=32))

    ds[0].BitsAllocated = 24
    ds[1].BitsAllocated = 24
    ds[2].BitsAllocated = 24

    self.assertRaises(InvalidDataset, numpy_grinder, ds)

  def test_numpy_SamplesPerPixel_3(self):
    ds = list(generate_numpy_datasets(3, Rows=2, Cols=2, rescale=False, Bits=32))

    ds[0].SamplesPerPixel = 3
    ds[1].SamplesPerPixel = 3
    ds[2].SamplesPerPixel = 3

    self.assertRaises(NotImplementedError, numpy_grinder, ds)

  def test_numpy_SamplesPerPixel_retired(self):
    ds = list(generate_numpy_datasets(3, Rows=2, Cols=2, rescale=False, Bits=32))

    ds[0].SamplesPerPixel = 4
    ds[1].SamplesPerPixel = 4
    ds[2].SamplesPerPixel = 4

    self.assertRaises(InvalidDataset, numpy_grinder, ds)

  def test_numpy_SamplesPerPixel_Nonsense(self):
    ds = list(generate_numpy_datasets(3, Rows=2, Cols=2, rescale=False, Bits=32))

    ds[0].SamplesPerPixel = 12
    ds[1].SamplesPerPixel = 12
    ds[2].SamplesPerPixel = 12

    self.assertRaises(InvalidDataset, numpy_grinder, ds)

