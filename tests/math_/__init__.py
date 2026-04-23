# Python standard library
from unittest import skipIf

# Third party modules
import numpy
import nibabel

# Dicomnode modules
from dicomnode import library_paths
from dicomnode.math import switch_ordering, CUDA
from dicomnode.math.image import Image
from dicomnode.math.space import Space

# Test modules
from tests.helpers.dicomnode_test_case import DicomnodeTestCase

ct_image_path = library_paths.report_data_directory / "CT_nifti" / "CT.nii"
ct_brain_path = library_paths.report_data_directory / "CT_nifti" / "segmentation" / "brain.nii.gz"

class MathTestCases(DicomnodeTestCase):
  def test_row_to_column(self):
    shape = (4,3,2)

    input_array = numpy.arange(numpy.prod(shape)).reshape(shape)
    test_array = switch_ordering(input_array)

    self.assertEqual(test_array.shape, tuple(reversed(shape)))
    to_list = [[[ elem for elem in subsublist] # Convert to build-in lists
                for subsublist in sublist]
                for sublist in test_array ]
    self.assertListEqual(to_list, [
      [[ 0.0,  6.0,  12.0, 18.0],
       [ 2.0,  8.0, 14.0, 20.0],
       [ 4.0, 10.0, 16.0, 22.0],
      ],
      [[ 1.0,  7.0, 13.0, 19.0],
       [ 3.0,  9.0, 15.0, 21.0],
       [ 5.0, 11.0, 17.0, 23.0],
      ]
    ])

  @skipIf(not CUDA, "You need GPU for this test")
  def test_center_of_gravity(self):
    image_data =  numpy.array([
      [[1,0,0,1],
       [0,0,0,0],
       [0,0,0,0],
       [1,0,0,1]],

      [[0,0,0,0],
       [0,0,0,0],
       [0,0,0,0],
       [0,0,0,0]],

      [[0,0,0,0],
       [0,0,0,0],
       [0,0,0,0],
       [0,0,0,0]],

      [[1,0,0,1],
       [0,0,0,0],
       [0,0,0,0],
       [1,0,0,1]],
    ], dtype=numpy.float32)

    image = Image(
      image_data, Space(
        numpy.eye(3), [0,0,0], image_data.shape
      )
    )

    from dicomnode.math import _cuda
    success, cog = _cuda.center_of_gravity(image.raw)

    self.assertEqual(cog[0], 1.5)
    self.assertEqual(cog[1], 1.5)
    self.assertEqual(cog[2], 1.5)


from . import tests_affine
from . import tests_bounding_box
from . import tests_mirror
from . import tests_image
from . import tests_types
from . import tests_space
from . import tests_gpu_helpers