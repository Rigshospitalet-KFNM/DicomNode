# Python Standard library
from unittest import TestCase

# Third party modules
import numpy

# Dicomnode modules
from dicomnode.math.image import fit_image_into_unsigned_bit_range, Image

# Test helper modules
from tests.helpers import generate_numpy_datasets

class ImageTestCase(TestCase):
  def test_fit_image_into_unsigned_bit_range_normal_use(self):
    data =  numpy.random.normal(0, 1, (50,50))
    min_val = data.min()
    max_val = data.max()

    bits_stored = 12
    bits_max_val = (1 << bits_stored) - 1

    fitted_data, slope, intercept = fit_image_into_unsigned_bit_range(
      data, bits_stored=bits_stored, bits_allocated=16
    )

    fitted_min = fitted_data.min()
    fitted_max = fitted_data.max()

    self.assertEqual(min_val, intercept)
    self.assertLess(bits_max_val - fitted_max, 5)
    self.assertEqual(fitted_min, 0)
    self.assertAlmostEqual(slope * bits_max_val + intercept, max_val)

  def test_fit_zeroes_into_unsigned_bit_range_normal_use(self):
    data = numpy.zeros((50,50)) + 100
    min_val = data.min()

    bits_stored = 12

    fitted_data, slope, intercept = fit_image_into_unsigned_bit_range(
      data, bits_stored=bits_stored, bits_allocated=16
    )

    fitted_min = fitted_data.min()
    fitted_max = fitted_data.max()

    self.assertEqual(min_val, intercept)
    self.assertEqual(fitted_max, 0)
    self.assertEqual(fitted_min, 0)
    self.assertEqual(slope, 1)

  def test_build_image(self):
    image = Image.from_datasets([ds for ds in generate_numpy_datasets(
      10, Cols=10, Rows=10
    )])

    self.assertEqual(image.raw.shape, (10,10,10))
    expected_affine = numpy.array([
      [4, 0, 0],
      [0, 4, 0],
      [0, 0, 4],
    ])

    self.assertTrue((image.space.basis==expected_affine).all())