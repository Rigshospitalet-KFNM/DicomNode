import numpy

from pydicom import Dataset
from pydicom.uid import SecondaryCaptureImageStorage
from unittest import TestCase, skip
from typing import List

from dicomnode.lib.dicom_factory import SOP_common_blueprint, FillingStrategy, image_plane_blueprint
from dicomnode.lib.numpy_factory import image_pixel_blueprint, NumpyFactory
from dicomnode.lib.exceptions import InvalidDataset


class NumpyFactoryTestCase(TestCase):
  def setUp(self) -> None:
    self.blueprint = image_pixel_blueprint + SOP_common_blueprint
    self.factory = NumpyFactory()
    self.header_dataset = Dataset()
    self.header_dataset.SOPClassUID = SecondaryCaptureImageStorage
    self.header = self.factory.make_series_header([self.header_dataset], self.blueprint)

  def test_make_series_no_encoding(self):
    images  = 100
    rows    = 40
    columns = 50
    image = numpy.random.randint(0, 65536, size=(images,rows,columns), dtype=numpy.uint16)

    datasets = self.factory.build_from_header(self.header, image)

    self.assertEqual(len(datasets), images)
    for dataset in datasets:
      self.assertEqual(dataset.SamplesPerPixel, 1)
      self.assertEqual(dataset.Columns, columns)
      self.assertEqual(dataset.Rows, rows)

  def test_make_series_float_encoding(self):
    images  = 100
    rows    = 40
    columns = 50
    image = numpy.random.uniform(
        -10e9,
        10e9,
        size=(images,rows,columns))

    datasets = self.factory.build_from_header(self.header, image)

    self.assertEqual(len(datasets), images)
    for dataset in datasets:
      self.assertEqual(dataset.Columns, columns)
      self.assertEqual(dataset.Rows, rows)

  def test_scale_image(self):
    image = numpy.array([[1.,2.], [3.,4.]], dtype=numpy.float64)
    scaled_image, slope, intercept = self.factory.scale_image(image)

    recreated_image = slope * scaled_image + intercept

    self.assertEqual(scaled_image.dtype, numpy.uint16)
    self.assertTrue(abs(scaled_image.max() - 65535) < 3)
    self.assertEqual(scaled_image.min(), 0)

    for image_val, recreated_val in zip(image.flatten(), recreated_image.flatten()):
      self.assertAlmostEqual(image_val, recreated_val, places=8)

  def test_scaled_negative_image(self):
    image = numpy.array([[-1024.,2.], [3.,4.]], dtype=numpy.float64)
    scaled_image, slope, intercept = self.factory.scale_image(image)

    recreated_image = slope * scaled_image + intercept

    self.assertEqual(scaled_image.dtype, numpy.uint16)
    self.assertTrue(abs(scaled_image.max() - 65535) < 3)
    self.assertEqual(scaled_image.min(), 0)

    for image_val, recreated_val in zip(image.flatten(), recreated_image.flatten()):
      self.assertAlmostEqual(image_val, recreated_val, places=1) # I have no idea but numerical unstably

  def test_numpy_factory_properties(self):
    factory = NumpyFactory()

    try:
      factory.pixel_representation = 3
      self.assertFalse(True)
    except ValueError:
      pass

    try:
      factory.pixel_representation = "asdf"
      self.assertFalse(True)
    except TypeError:
      pass

    factory.pixel_representation = 1
    self.assertEqual(factory.pixel_representation, 1)

    try:
      factory.bits_allocated = "asdf" #type: ignore the point of this test
      self.assertFalse(True)
    except TypeError:
      pass

    try:
      factory.bits_allocated = 17
      self.assertFalse(True)
    except ValueError:
      pass

    try:
      factory.bits_allocated = -16
      self.assertFalse(True)
    except ValueError:
      pass

    factory.bits_allocated = 24
    self.assertEqual(factory.bits_allocated, 24)

    try:
      factory.bits_stored = "asdf" #type: ignore the point of this test
      self.assertFalse(True)
    except TypeError:
      pass

    try:
      factory.bits_stored = -16
      self.assertFalse(True)
    except ValueError:
      pass

    try:
      factory.bits_stored = 0
      self.assertFalse(True)
    except ValueError:
      pass

    factory.bits_stored = 1
    self.assertEqual(factory.bits_stored, 1)
    factory.bits_stored = 20
    self.assertEqual(factory.bits_stored, 20)
    factory.bits_stored = 24
    self.assertEqual(factory.bits_stored, 24)

    try:
      factory.high_bit = "asdf" #type: ignore the point of this test
      self.assertFalse(True)
    except TypeError:
      pass

    try:
      factory.high_bit = -16
      self.assertFalse(True)
    except ValueError:
      pass

    try:
      factory.high_bit = 0
      self.assertFalse(True)
    except ValueError:
      pass

    factory.high_bit = 23

  # Dicom factory

  def test_SOP_common_header(self):
    dataset = Dataset()
    dataset.SOPClassUID = SecondaryCaptureImageStorage
    blueprint = self.blueprint + SOP_common_blueprint

    factory = NumpyFactory()
    header = factory.make_series_header([dataset], blueprint, FillingStrategy.DISCARD)

    images  = 2
    rows    = 40
    columns = 50
    image = numpy.random.randint(0, 65536, size=(images,rows,columns), dtype=numpy.uint16)

    datasets = factory.build_from_header(header, image)

    for i, ds in enumerate(datasets):
      self.assertEqual(ds.SOPClassUID, SecondaryCaptureImageStorage)
      self.assertIn(0x00080018, ds)
      self.assertEqual(ds.InstanceNumber, i + 1)
