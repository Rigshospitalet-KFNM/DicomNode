# Python Standard library
from copy import deepcopy
from unittest import TestCase

# Third party modules
import numpy

# Dicomnode modules
from dicomnode.math.types import MirrorDirection
from dicomnode.math.space import Space, ReferenceSpace
from dicomnode.math.image import fit_image_into_unsigned_bit_range, Image

# Test helper modules
from tests.helpers import generate_numpy_datasets

class ImageTestCase(TestCase):
  def setUp(self):
    self.ras_image_data = numpy.array([
      [
        [1,0,0,2],
        [0,0,0,0],
        [0,0,0,0],
        [3,0,0,4]
      ],
      [
        [5,0,0,6],
        [0,0,0,0],
        [0,0,0,0],
        [7,0,0,8]
      ],
      [
        [9,0,0,10],
        [0,0,0,0],
        [0,0,0,0],
        [11,0,0,12]
      ],
      [
        [13,0,0,14],
        [0,0,0,0],
        [0,0,0,0],
        [15,0,0,16]
      ],
    ])

    self.space_ras = Space([  # type: ignore
      [4,0,0],
      [0,4,0],
      [0,0,4]
    ], [0,0,0], [4,4,4])

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

  def test_image_directional_irrelevant_instance_number(self):
    datasets = [ds for ds in generate_numpy_datasets(10, Cols=10, Rows=10)]
    copy_datasets = deepcopy(datasets)

    for i,dataset in enumerate(copy_datasets):
      dataset.InstanceNumber = len(datasets) - i

    image = Image.from_datasets(datasets)
    copy_image = Image.from_datasets(copy_datasets)

    self.assertTrue((image.raw == copy_image.raw).all())

  def test_image_mirroring_ras_to_ras(self):
    """Tests if mirroring is done correctly"""

    image_ras = Image(deepcopy(self.ras_image_data), deepcopy(self.space_ras)) #type: ignore

    image_ras.transform_to_ras()

    self.assertEqual(self.space_ras, image_ras.space)
    self.assertTrue((image_ras.raw == self.ras_image_data).all())

  def test_image_mirroring_las_to_ras(self):
    image_data_las = numpy.array([
      [
        [2,0,0,1],
        [0,0,0,0],
        [0,0,0,0],
        [4,0,0,3]
      ],
      [
        [6,0,0,5],
        [0,0,0,0],
        [0,0,0,0],
        [8,0,0,7]
      ],
      [
        [10,0,0,9],
        [0,0,0,0],
        [0,0,0,0],
        [12,0,0,11]
      ],
      [
        [14,0,0,13],
        [0,0,0,0],
        [0,0,0,0],
        [16,0,0,15]
      ],
    ])

    space_las = Space([  # type: ignore
      [-4,0,0],
      [0,4,0],
      [0,0,4]
    ], [16,0,0], [4,4,4])

    image_las = Image(image_data_las, space_las)
    self.assertEqual(image_las.space.reference_space, ReferenceSpace.LAS)

    image_las.transform_to_ras()

    self.assertEqual(image_las.space.reference_space, ReferenceSpace.RAS)
    self.assertEqual(self.space_ras, image_las.space)
    self.assertTrue((image_las.raw == self.ras_image_data).all())

  def test_image_mirroring_lps_to_ras(self):
    image_data_lps = numpy.array([
      [
        [4,0,0,3],
        [0,0,0,0],
        [0,0,0,0],
        [2,0,0,1],
      ],
      [
        [8,0,0,7],
        [0,0,0,0],
        [0,0,0,0],
        [6,0,0,5],
      ],
      [
        [12,0,0,11],
        [0,0,0,0],
        [0,0,0,0],
        [10,0,0,9],
      ],
      [
        [16,0,0,15],
        [0,0,0,0],
        [0,0,0,0],
        [14,0,0,13],
      ],
    ])

    space_lps = Space([  # type: ignore
      [-4,0,0],
      [0,-4,0],
      [0,0,4]
    ], [16,16,0], [4,4,4])

    image_las = Image(image_data_lps, space_lps)
    self.assertEqual(image_las.space.reference_space, ReferenceSpace.LPS)

    image_las.transform_to_ras()

    self.assertEqual(image_las.space.reference_space, ReferenceSpace.RAS)
    self.assertEqual(self.space_ras, image_las.space)
    self.assertTrue((image_las.raw == self.ras_image_data).all())

  def test_image_mirroring_lpi_to_ras(self):
    image_data_lpi = numpy.array([
      [
        [16,0,0,15],
        [0,0,0,0],
        [0,0,0,0],
        [14,0,0,13],
      ],
      [
        [12,0,0,11],
        [0,0,0,0],
        [0,0,0,0],
        [10,0,0,9],
      ],
      [
        [8,0,0,7],
        [0,0,0,0],
        [0,0,0,0],
        [6,0,0,5],
      ],
      [
        [4,0,0,3],
        [0,0,0,0],
        [0,0,0,0],
        [2,0,0,1],
      ],
    ])

    space_lpi = Space([  # type: ignore
      [-4,0,0],
      [0,-4,0],
      [0,0,-4]
    ], [16,16,16], [4,4,4])

    image_las = Image(image_data_lpi, space_lpi)
    self.assertEqual(image_las.space.reference_space, ReferenceSpace.LPI)

    image_las.transform_to_ras()

    self.assertEqual(image_las.space.reference_space, ReferenceSpace.RAS)
    self.assertEqual(self.space_ras, image_las.space)
    self.assertTrue((image_las.raw == self.ras_image_data).all())

  def test_image_mirroring_lai_to_ras(self):
    image_data_lai = numpy.array([
      [
        [14,0,0,13],
        [0,0,0,0],
        [0,0,0,0],
        [16,0,0,15]
      ],
      [
        [10,0,0,9],
        [0,0,0,0],
        [0,0,0,0],
        [12,0,0,11]
      ],
      [
        [6,0,0,5],
        [0,0,0,0],
        [0,0,0,0],
        [8,0,0,7]
      ],
      [
        [2,0,0,1],
        [0,0,0,0],
        [0,0,0,0],
        [4,0,0,3]
      ],
    ])

    space_lai = Space([  # type: ignore
      [-4,0,0],
      [0,4,0],
      [0,0,-4]
    ], [16,0,16], [4,4,4])

    image_las = Image(image_data_lai, space_lai)
    self.assertEqual(image_las.space.reference_space, ReferenceSpace.LAI)

    image_las.transform_to_ras()

    self.assertEqual(image_las.space.reference_space, ReferenceSpace.RAS)
    self.assertEqual(self.space_ras, image_las.space)
    self.assertTrue((image_las.raw == self.ras_image_data).all())

  def test_image_mirroring_rai_to_ras(self):
    image_data_rai = numpy.array([
      [
        [13,0,0,14],
        [0,0,0,0],
        [0,0,0,0],
        [15,0,0,16]
      ],
      [
        [9,0,0,10],
        [0,0,0,0],
        [0,0,0,0],
        [11,0,0,12]
      ],
      [
        [5,0,0,6],
        [0,0,0,0],
        [0,0,0,0],
        [7,0,0,8]
      ],
      [
        [1,0,0,2],
        [0,0,0,0],
        [0,0,0,0],
        [3,0,0,4]
      ],
    ])

    space_rai = Space([  # type: ignore
      [4,0,0],
      [0,4,0],
      [0,0,-4]
    ], [0,0,16], [4,4,4])

    image_las = Image(image_data_rai, space_rai)
    self.assertEqual(image_las.space.reference_space, ReferenceSpace.RAI)

    image_las.transform_to_ras()

    self.assertEqual(image_las.space.reference_space, ReferenceSpace.RAS)
    self.assertEqual(self.space_ras, image_las.space)
    self.assertTrue((image_las.raw == self.ras_image_data).all())

  def test_image_mirroring_rpi_to_ras(self):
    image_data_rpi = numpy.array([
      [
        [15,0,0,16],
        [0,0,0,0],
        [0,0,0,0],
        [13,0,0,14],
      ],
      [
        [11,0,0,12],
        [0,0,0,0],
        [0,0,0,0],
        [9,0,0,10],
      ],
      [
        [7,0,0,8],
        [0,0,0,0],
        [0,0,0,0],
        [5,0,0,6],
      ],
      [
        [3,0,0,4],
        [0,0,0,0],
        [0,0,0,0],
        [1,0,0,2],
      ],
    ])

    space_rpi = Space([  # type: ignore
      [4,0,0],
      [0,-4,0],
      [0,0,-4]
    ], [0,16,16], [4,4,4])

    image_rpi = Image(image_data_rpi, space_rpi)
    self.assertEqual(image_rpi.space.reference_space, ReferenceSpace.RPI)

    image_rpi.transform_to_ras()

    self.assertEqual(image_rpi.space.reference_space, ReferenceSpace.RAS)
    self.assertEqual(self.space_ras, image_rpi.space)
    self.assertTrue((image_rpi.raw == self.ras_image_data).all())

  def test_image_mirroring_rps_to_ras(self):
    image_data_rps = numpy.array([
      [
        [3,0,0,4],
        [0,0,0,0],
        [0,0,0,0],
        [1,0,0,2],
      ],
      [
        [7,0,0,8],
        [0,0,0,0],
        [0,0,0,0],
        [5,0,0,6],
      ],
      [
        [11,0,0,12],
        [0,0,0,0],
        [0,0,0,0],
        [9,0,0,10],
      ],
      [
        [15,0,0,16],
        [0,0,0,0],
        [0,0,0,0],
        [13,0,0,14],
      ],
    ])

    space_rps = Space([  # type: ignore
      [4,0,0],
      [0,-4,0],
      [0,0,4]
    ], [0,16,0], [4,4,4])

    image_rps = Image(image_data_rps, space_rps)
    self.assertEqual(image_rps.space.reference_space, ReferenceSpace.RPS)

    image_rps.transform_to_ras()

    self.assertEqual(image_rps.space.reference_space, ReferenceSpace.RAS)
    self.assertEqual(self.space_ras, image_rps.space)
    self.assertTrue((image_rps.raw == self.ras_image_data).all())