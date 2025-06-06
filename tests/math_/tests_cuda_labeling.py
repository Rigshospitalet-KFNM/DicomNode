# Python standard library
from unittest import skipIf

# Third party modules
import numpy

# Dicomnode modules
from dicomnode.math import CUDA
from dicomnode.math.image import Image
from dicomnode.math.space import Space
from dicomnode.math import labeling

from tests.helpers.dicomnode_test_case import DicomnodeTestCase

class LabelingTestCase(DicomnodeTestCase):
  @skipIf(not CUDA, "Need GPU for testcase")
  def test_GPU_labeling_uint32(self):
    space = Space(
      basis=numpy.array(
        [[1.0,0.0,0.0],
         [0.0,1.0,0.0],
         [0.0,0.0,1.0],
        ]
    ),
      starting_point=numpy.array([1,1,1]),
      domain=numpy.array([3,3,3]))

    image = Image(numpy.array([
      [
        [1,1,1],
        [1,0,1],
        [1,1,1]
      ],
      [
        [1,1,1],
        [0,0,0],
        [1,1,1]
      ],
      [
        [1,0,1],
        [0,1,0],
        [1,0,1]
      ]
    ], dtype=numpy.uint32), space)

    cuda_success, res = labeling._gpu_labeling(image)

    from dicomnode.math import _cuda

    self.assertIsInstance(cuda_success, _cuda.DicomnodeError)
    self.assertIsInstance(res, numpy.ndarray)

    expected_res = numpy.array([
      [
        [1,1,1],
        [1,0,1],
        [1,1,1]
      ],
      [
        [1,1,1],
        [0,0,0],
        [7,7,7]
      ],
      [
        [1,0,1],
        [0,1,0],
        [1,0,1]
      ]
    ])


    self.assertTrue((res == expected_res).all())

  @skipIf(not CUDA, "Need GPU for testcase")
  def test_GPU_labeling_uint8(self):
    space = Space(
      basis=numpy.array(
        [[1.0,0.0,0.0],
         [0.0,1.0,0.0],
         [0.0,0.0,1.0],
        ]
    ),
      starting_point=numpy.array([1,1,1]),
      domain=numpy.array([3,3,3]))

    image = Image(numpy.array([
      [
        [1,1,1],
        [1,0,1],
        [1,1,1]
      ],
      [
        [1,1,1],
        [0,0,0],
        [1,1,1]
      ],
      [
        [1,0,1],
        [0,1,0],
        [1,0,1]
      ]
    ], dtype=numpy.uint8), space)

    cuda_success, res = labeling._gpu_labeling(image)

    from dicomnode.math import _cuda

    self.assertIsInstance(cuda_success, _cuda.DicomnodeError)
    self.assertIsInstance(res, numpy.ndarray)
    self.assertEqual(numpy.dtype(dtype=numpy.uint32), res.dtype)

    expected_res = numpy.array([
      [
        [1,1,1],
        [1,0,1],
        [1,1,1]
      ],
      [
        [1,1,1],
        [0,0,0],
        [7,7,7]
      ],
      [
        [1,0,1],
        [0,1,0],
        [1,0,1]
      ]
    ])


    self.assertTrue((res == expected_res).all())
