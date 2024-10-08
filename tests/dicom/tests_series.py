"""Tests src/dicomnode/dicom/series.py"""

# Python standard library
from typing import List
from unittest import TestCase

# Third party module
from numpy import ndarray
from pydicom import Dataset, DataElement

# Dicomnode
from dicomnode.dicom import gen_uid
from dicomnode.dicom.series import DicomSeries, NiftiSeries, sortDatasets, shared_tag
from dicomnode.math.image import Image

# Test stuff
from tests.helpers import generate_numpy_datasets

class DicomSeriesTestCase(TestCase):
  def setUp(self) -> None:
    self.num_datasets = 3

    self.datasets = [ds for ds in generate_numpy_datasets(
      self.num_datasets, Cols=3, Rows=3
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

class SharedTagsTestCase(TestCase):
  def test_empty_list(self):
    with self.assertRaises(ValueError):
      shared_tag([],0x0010_0010)
