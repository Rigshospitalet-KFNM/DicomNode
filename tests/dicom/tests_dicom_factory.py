"""Yes I know there's also tests in tests_dicom_dicom_factory"""

# Python standard library

# Third party modules
import numpy
from pydicom.uid import PositronEmissionTomographyImageStorage,\
  SecondaryCaptureImageStorage

# Dicomnode modules
from dicomnode.lib.exceptions import IncorrectlyConfigured
from dicomnode.math.image import Image
from dicomnode.math.space import Space
from dicomnode.dicom.dicom_factory import Blueprint, DicomFactory, StaticElement,\
  FunctionalElement, ImageOrientationElement, ImagePositionElement

# Test modules
from tests.helpers.dicomnode_test_case import DicomnodeTestCase

class DicomFactory4DSeriesTests(DicomnodeTestCase):
  def setUp(self) -> None:
    self.frames = 5 # t dim
    self.slices = 6 # z - dim
    self.rows = 7 # y - dim
    self.columns = 8 # x - dim

    # Build the Image
    self.image_data = numpy.random.uniform(
      0,1, (self.frames, self.slices, self.rows, self.columns)
    ).astype(numpy.float32)

    self.space = Space(
      3 * numpy.eye(3), [-10,-20,-30], (self.slices, self.rows, self.columns)
    )

    self.image =  Image(self.image_data, self.space)

  def test_dicom_factory_4d_series_missing_ClassUID_raises(self):
    factory = DicomFactory()

    blueprint = Blueprint([
      # Conspicuous lack of SOPInstanceUID
     ])
    self.assertRaises(IncorrectlyConfigured,factory.build_4d_series, self.image, blueprint)

  def test_dicom_factory_4d_series_dynamic_ClassUID_raises(self):
    factory = DicomFactory()

    blueprint = Blueprint([
      FunctionalElement(0x0008_0016, "UI", lambda ie: PositronEmissionTomographyImageStorage)
    ])
    self.assertRaises(IncorrectlyConfigured,factory.build_4d_series, self.image, blueprint)

  def test_dicom_factory_4d_series_unsupported_ClassUID_raises(self):
    factory = DicomFactory()

    blueprint = Blueprint([
      StaticElement(0x0008_0016, "UI", SecondaryCaptureImageStorage)
    ])

    self.assertRaises(IncorrectlyConfigured,factory.build_4d_series, self.image, blueprint)

  def test_dicom_factory_4d_series_pet_series_without_gated_or_dynamic(self):
    factory = DicomFactory()

    blueprint = Blueprint([
      StaticElement(0x0008_0016, "UI", PositronEmissionTomographyImageStorage)
    ])

    self.assertRaises(IncorrectlyConfigured,factory.build_4d_series, self.image, blueprint)

  def test_dicom_factory_4d_series_pet_series_misspelling(self):
    factory = DicomFactory()

    blueprint = Blueprint([
      StaticElement(0x0008_0016, "UI", PositronEmissionTomographyImageStorage)
    ])

    with self.assertRaises(IncorrectlyConfigured):
      factory.build_4d_series(self.image, blueprint, series_type="OOOOOOOOH WHAT LOVE GOT TO DO WITH IT")

  def test_dicom_factory_4d_series_pet_series_dynamic(self):
    factory = DicomFactory()

    blueprint = Blueprint([
      StaticElement(0x0008_0016, "UI", PositronEmissionTomographyImageStorage)
    ])

    series = factory.build_4d_series(self.image, blueprint, series_type="DYNAMIC")

    self.assertEqual(len(series), self.frames * self.slices)

    for dataset in series:
      self.assertIn(0x0054_0081, dataset)
      self.assertIn(0x0054_1000, dataset)
      self.assertIn(0x7FE0_0010, dataset)

  def test_dicom_factory_4d_series_pet_series_gated(self):
    factory = DicomFactory()

    blueprint = Blueprint([
      StaticElement(0x0008_0016, "UI", PositronEmissionTomographyImageStorage),
      ImagePositionElement(self.space),
      ImageOrientationElement(self.space)
    ])

    series = factory.build_4d_series(self.image, blueprint, series_type="GATED")

    self.assertEqual(len(series), self.frames * self.slices)

    for i, dataset in enumerate(series):
      self.assertIn(0x0054_0081, dataset)
      self.assertIn(0x0054_1000, dataset)
      self.assertIn(0x7FE0_0010, dataset)
      self.assertEqual(dataset.Rows, self.rows)
      self.assertEqual(dataset.Columns, self.columns)

      if self.slices < i:
        self.assertEqual(dataset.ImageOrientationPatient, series.datasets[i % self.slices].ImageOrientationPatient)
        self.assertEqual(dataset.ImagePositionPatient, series.datasets[i % self.slices].ImagePositionPatient)
