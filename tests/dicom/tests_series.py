"""Tests src/dicomnode/dicom/series.py"""

# Python standard library
from typing import List
from unittest import TestCase

# Third party module
from nibabel.nifti1 import Nifti1Image, Nifti1Header
import numpy
from pydicom import Dataset, DataElement
from pydicom.tag import Tag

# Dicomnode
from dicomnode.lib.exceptions import IncorrectlyConfigured
from dicomnode.dicom import gen_uid
from dicomnode.dicom.series import DicomSeries, NiftiSeries, shared_tag,\
  Series, extract_image
from dicomnode.math.image import Image

# Test stuff
from tests.helpers import generate_numpy_datasets

class DicomSeriesTestCase(TestCase):
  def setUp(self) -> None:
    self.num_datasets = 3

    self.datasets = [ds for ds in generate_numpy_datasets(
      self.num_datasets, Cols=3, Rows=3, PatientID="Blah"
    )]

  def test_create_empty_dicom_series(self):
    with self.assertRaises(ValueError):
      DicomSeries([])

  def test_create_dicom_series_iter(self):
    with self.assertNoLogs('dicomnode'):
      ds = DicomSeries(self.datasets)

    for series_set, dataset in zip(ds, self.datasets):
      self.assertIs(series_set,dataset)

  def test_create_image_from_series(self):
    ds = DicomSeries(self.datasets)
    image = ds.image
    self.assertIsInstance(image, Image)

  def test_dicomnode_get_tag(self):
    ds = DicomSeries(self.datasets)
    de = ds[0x0028_0010]
    if not isinstance(de, DataElement):
      self.assertIsInstance(de, DataElement)
      raise Exception
    self.assertEqual(de.value, 3)
    sop_instance_uids = ds[0x0008_0018]
    self.assertIsInstance(sop_instance_uids, List)
    if not isinstance(sop_instance_uids, List):
      self.assertFalse(True)
      raise Exception

    self.assertEqual(len(sop_instance_uids), self.num_datasets)

  def test_shared_tag(self):
    ds = DicomSeries(self.datasets)
    self.assertTrue(ds.shared_tag(0x0028_0010))
    self.assertFalse(ds.shared_tag(0x0008_0018))


  def test_set_tag_shared(self):
    ds = DicomSeries(self.datasets)
    patient_name ="Test username"
    ds[0x0010_0010] = patient_name

    for dataset in ds:
      self.assertIn(0x0010_0010, dataset)
      self.assertEqual(dataset[0x0010_0010].value, patient_name)

  def test_set_individual_tag(self):
    ds = DicomSeries(self.datasets)

    ds[0x0008_0018] = [gen_uid() for _ in range(self.num_datasets)]

    for dataset in ds:
      self.assertIn(0x0008_0018, dataset)

  def test_set_faulty_test(self):
    ds = DicomSeries(self.datasets)

    with self.assertRaises(ValueError):
      ds[0x0008_0018] = [gen_uid(), gen_uid()]

    with self.assertRaises(ValueError):
      ds.set_individual_tag(0x0008_0018, [gen_uid(), gen_uid()])

    with self.assertRaises(TypeError):
      ds[0x0008_0018] = gen_uid()

  def test_get_tag_attribute_from_series(self):
    ds = DicomSeries(self.datasets)
    self.assertEqual(ds.PatientID, "Blah")

  def test_series_iterable(self):
    ds = DicomSeries(self.datasets)

    for dataset in ds:
      self.assertIsInstance(dataset, Dataset)
      self.assertIn(dataset, self.datasets)

  def test_setting_shared_attribute_with_string_key(self):
    ds = DicomSeries(self.datasets)

    ds["PatientName"] = "Hello world"

    for dataset in ds:
      self.assertEqual(dataset.PatientName, "Hello world")


class SharedTagsTestCase(TestCase):
  def test_empty_list(self):
    with self.assertRaises(ValueError):
      shared_tag([], Tag(0x0010_0010))

class BaseSeriesTestCase(DicomSeriesTestCase):
  def test_fault_constructor(self):
    series = Series(None) # type: ignore
    with self.assertRaises(IncorrectlyConfigured):
      series.image

class NiftiSeriesTestCase(DicomSeriesTestCase):
  def test_nifti(self):
    shape = (7,6,5)
    header = Nifti1Header()
    data = numpy.asfortranarray(numpy.random.random(shape))
    affine = numpy.array([
      [3.0, 0.0, 0.0, 10.0],
      [0.0, 3.0, 0.0, 20.0],
      [0.0, 0.0, 4.0, 30.0],
      [0.0, 0.0, 0.0, 1.0]
    ])
    image = Nifti1Image(data, affine, header)

    series = NiftiSeries(image)

    self.assertEqual(series.image.shape[0], 5)
    self.assertEqual(series.image.shape[1], 6)
    self.assertEqual(series.image.shape[2], 7)

    self.assertEqual(series.image.space.starting_point[0], [10])
    self.assertEqual(series.image.space.starting_point[1], [20])
    self.assertEqual(series.image.space.starting_point[2], [30])

class SeriesUtilitiesTestCases(DicomSeriesTestCase):
  def test_extract_image(self):
    series = DicomSeries([ds for ds in generate_numpy_datasets(
      10, Cols=3, Rows=3, PatientID="Blah", pixel_spacing=[1,1],
      starting_image_position=[0,0,0], image_orientation=[1,0,0,0,1,0]
    )])

    self.assertIs(extract_image(series), series.image)
    self.assertIs(extract_image(series.image), series.image)

  def test_extract_from_nonsense(self):
    self.assertRaises(TypeError, extract_image, 1)

  def test_from_dataset_list(self):
    datasets = [ds for ds in generate_numpy_datasets(
      10, Cols=3, Rows=3, PatientID="Blah", pixel_spacing=[1,1],
      starting_image_position=[0,0,0], image_orientation=[1,0,0,0,1,0]
    )]

    self.assertIsInstance(extract_image(datasets),Image)