"""Tests src/dicomnode/dicom/series.py"""

# Python standard library
from datetime import date, time
from logging import getLogger
from typing import List
from unittest import TestCase

# Third party module
from nibabel.nifti1 import Nifti1Image, Nifti1Header
import numpy
from pydicom import Dataset, DataElement
from pydicom.tag import Tag

# Dicomnode
from dicomnode.constants import DICOMNODE_LOGGER_NAME

from dicomnode.lib.exceptions import IncorrectlyConfigured,\
  MissingPivotDataset, InvalidDataset
from dicomnode.dicom import gen_uid
from dicomnode.dicom.series import DicomSeries, NiftiSeries, shared_tag,\
  Series, extract_image, frame_unrelated_series,\
  extract_space
from dicomnode.math.image import Image
from dicomnode.math.space import Space
from dicomnode.lib.utils import is_picklable

# Test stuff
from tests.helpers import generate_numpy_datasets, generate_dummy_space
from tests.helpers.dicomnode_test_case import DicomnodeTestCase

class SeriesTestCase(DicomnodeTestCase):
  def test_series_generate_2_dimensional_slices(self):
    shape_3d = (4,5,5)
    shape_4d = (3,4,5,5)
    shape_5d = (2,3,4,5,5)

    data_3d = numpy.random.uniform(0,1, shape_3d)
    data_4d = numpy.random.uniform(0,1, shape_4d)
    data_5d = numpy.random.uniform(0,1, shape_5d)

    space_3d = generate_dummy_space(shape_3d)
    space_4d = generate_dummy_space(shape_4d)
    space_5d = generate_dummy_space(shape_5d)

    series_3d = Series(Image(data_3d, space_3d))
    series_4d = Series(Image(data_4d, space_4d))
    series_5d = Series(Image(data_5d, space_5d))

    self.assertEqual(series_3d.image.number_frames(), 1)
    for index_3d, slice_ in enumerate(series_3d.slices()):
      self.assertEqual(slice_.ndim, 2)
      self.assertTrue((series_3d.image.raw[index_3d] == slice_).all())

    self.assertEqual(series_4d.image.number_frames(), 3)
    for i, slice_ in enumerate(series_4d.slices()):
      index_3d = i % shape_4d[1]
      index_4d = i // shape_4d[1]

      self.assertEqual(slice_.ndim, 2)
      self.assertTrue((series_4d.image.raw[index_4d, index_3d] == slice_).all())

    self.assertEqual(series_5d.image.number_frames(), 6)
    for i, slice_ in enumerate(series_5d.slices()):
      index_3d = i % shape_5d[2]
      index_4d = (i // shape_5d[2]) % shape_5d[1]
      index_5d = i // (shape_5d[2] * shape_5d[1])

      self.assertEqual(slice_.ndim, 2)
      self.assertTrue((series_5d.image.raw[index_5d, index_4d, index_3d] == slice_).all())

class DicomSeriesTestCase(DicomnodeTestCase):
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

  def test_empty_list(self):
    with self.assertRaises(ValueError):
      shared_tag([], Tag(0x0010_0010))

  def test_can_datasets_house_image(self):
    series = DicomSeries([Dataset() for _ in range(10)])

    fitting_image = numpy.empty((10, 40, 40))

    self.assertTrue(series.can_copy_into_image(fitting_image))

    unfitting_image = numpy.empty((20, 40, 40))

    self.assertFalse(series.can_copy_into_image(unfitting_image))

  def test_fault_constructor(self):
    series = Series(None) # type: ignore
    with self.assertRaises(IncorrectlyConfigured):
      series.image

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

  def test_extract_image(self):
    series = DicomSeries([ds for ds in generate_numpy_datasets(
      10, Cols=3, Rows=3, PatientID="Blah", pixel_spacing=[1,1],
      starting_image_position=[0,0,0], image_orientation=[1,0,0,0,1,0]
    )])

    self.assertIs(extract_image(series), series.image)
    self.assertIs(extract_image(series.image), series.image)

    self.assertIsInstance(
      extract_image(
        self.create_dynamic_pet_series()
      ),
      Image)


  def test_from_dataset_list(self):
    datasets = [ds for ds in generate_numpy_datasets(
      10, Cols=3, Rows=3, PatientID="Blah", pixel_spacing=[1,1],
      starting_image_position=[0,0,0], image_orientation=[1,0,0,0,1,0]
    )]

    self.assertIsInstance(extract_image(datasets),Image)

  def create_dynamic_pet_series(self):
    number_of_slices_per_series = 10
    frame_1 = DicomSeries([ds for ds in generate_numpy_datasets(
      number_of_slices_per_series, Cols=3, Rows=3, PatientID="Blah", pixel_spacing=[1,1],
      starting_image_position=[0,0,0], image_orientation=[1,0,0,0,1,0],
      slice_thickness=3
    )])
    frame_2 = DicomSeries([ds for ds in generate_numpy_datasets(
      number_of_slices_per_series, Cols=3, Rows=3, PatientID="Blah", pixel_spacing=[1,1],
      starting_image_position=[0,0,0], image_orientation=[1,0,0,0,1,0],
      slice_thickness=3
    )])
    frame_3 = DicomSeries([ds for ds in generate_numpy_datasets(
      number_of_slices_per_series, Cols=3, Rows=3, PatientID="Blah", pixel_spacing=[1,1],
      starting_image_position=[0,0,0], image_orientation=[1,0,0,0,1,0],
      slice_thickness=3
    )])

    frames = [frame_1, frame_2, frame_3]
    num_of_frames = len(frames)

    frame_durations = [1000, 1000, 1000]

    for i, frame in enumerate(frames):
      frame["NumberOfTimeSlices"] = num_of_frames
      frame["ImageIndex"] = [i * number_of_slices_per_series + j for j in range(number_of_slices_per_series)]
      frame["ActualFrameDuration"] = frame_durations[i]
      frame["AcquisitionDate"] = date(2020, 5, 20)
      frame["AcquisitionTime"] = time(10, 14, 13 + i, 63122)

    return frame_unrelated_series(*frames)

  def test_create_dynamic_pet_series(self):
    slices_per_series = 5
    frame_1 = DicomSeries([ds for ds in generate_numpy_datasets(
      slices_per_series, Cols=3, Rows=3, PatientID="Blah", pixel_spacing=[1,1],
      starting_image_position=[0,0,0], image_orientation=[1,0,0,0,1,0],
      slice_thickness=3
    )])
    frame_2 = DicomSeries([ds for ds in generate_numpy_datasets(
      slices_per_series, Cols=3, Rows=3, PatientID="Blah", pixel_spacing=[1,1],
      starting_image_position=[0,0,0], image_orientation=[1,0,0,0,1,0],
      slice_thickness=3
    )])
    frame_3 = DicomSeries([ds for ds in generate_numpy_datasets(
      slices_per_series, Cols=3, Rows=3, PatientID="Blah", pixel_spacing=[1,1],
      starting_image_position=[0,0,0], image_orientation=[1,0,0,0,1,0],
      slice_thickness=3
    )])

    frame_1_image = frame_1.image # Build the Images now otherwise they become 4 images
    frame_2_image = frame_2.image # Build the Images now otherwise they become 4 images
    frame_3_image = frame_3.image # Build the Images now otherwise they become 4 images

    frames = [frame_1, frame_2, frame_3]
    num_of_frames = len(frames)

    frame_durations = [1000, 1000, 1000]

    for i, frame in enumerate(frames):
      frame["NumberOfTimeSlices"] = num_of_frames
      frame["NumberOfSlices"] = slices_per_series
      frame["ImageIndex"] = [i * slices_per_series + j for j in range(slices_per_series)]
      frame["ActualFrameDuration"] = frame_durations[i]
      frame["AcquisitionDate"] = date(2020, 5, 20)
      frame["AcquisitionTime"] = time(10, 14, 13 + i, 63122)

    pet_series = frame_unrelated_series(*frames)

    self.assertIsInstance(pet_series, DicomSeries)
    self.assertIsInstance(pet_series.image, Image)
    self.assertIsInstance(pet_series.frame(0), Image)
    self.assertIsInstance(pet_series.image.raw, numpy.ndarray)
    self.assertEqual(pet_series.image.raw.dtype, numpy.float32)
    self.assertEqual(pet_series.image.raw.ndim, 4)

    self.assertTrue((pet_series.image.raw[0,:,:,:] == frame_1_image.raw).all())
    self.assertTrue((pet_series.image.raw[1,:,:,:] == frame_2_image.raw).all())
    self.assertTrue((pet_series.image.raw[2,:,:,:] == frame_3_image.raw).all())

  def test_unable_to_frame_no_images(self):
    self.assertRaises(ValueError, frame_unrelated_series)

  def test_unable_to_frame_uneven_number_of_images(self):
    frame_1 = DicomSeries([ds for ds in generate_numpy_datasets(
      10, Cols=3, Rows=3, PatientID="Blah", pixel_spacing=[1,1],
      starting_image_position=[0,0,0], image_orientation=[1,0,0,0,1,0],
      slice_thickness=3
    )])
    frame_2 = DicomSeries([ds for ds in generate_numpy_datasets(
      11, Cols=3, Rows=3, PatientID="Blah", pixel_spacing=[1,1],
      starting_image_position=[0,0,0], image_orientation=[1,0,0,0,1,0],
      slice_thickness=3
    )])

    self.assertRaises(ValueError, frame_unrelated_series, frame_1, frame_2)

  def test_extract_image_type_error_on_bogus_arg(self):
    with self.assertLogs(DICOMNODE_LOGGER_NAME):
      self.assertRaises(TypeError, extract_image, 1)

  def test_extract_space(self):
    self.assertRaises(TypeError, extract_space, 1)

    series = DicomSeries([ds for ds in generate_numpy_datasets(
      10, Cols=3, Rows=3, PatientID="Blah", pixel_spacing=[1,1],
      starting_image_position=[0,0,0], image_orientation=[1,0,0,0,1,0],
      slice_thickness=3
    )])

    pet_series = self.create_dynamic_pet_series()

    self.assertIsInstance(extract_space(pet_series), Space)
    self.assertIsInstance(extract_space(pet_series.image), Space)
    self.assertIsInstance(extract_space(pet_series.frame(1)), Space)
    self.assertIsInstance(extract_space(series), Space)
    self.assertIsInstance(extract_space(series.datasets), Space)

  def test_dicom_series_is_pickleable(self):
    self.assertTrue(is_picklable(
      DicomSeries([ds for ds in generate_numpy_datasets(
        10, Cols=3, Rows=3, PatientID="Blah", pixel_spacing=[1,1],
        starting_image_position=[0,0,0], image_orientation=[1,0,0,0,1,0],
        slice_thickness=3
    )])
    ))

  def test_dicom_series_count_frame_and_slices(self):
    series = self.create_dynamic_pet_series()

    self.assertEqual(series.image.number_frames(), 3)
    self.assertEqual(series.image.number_slices(), 30)