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
  def test_GPU_labeling_int(self):
    space = Space(
      basis=numpy.array(
        [[1.0,0.0,0.0],
         [0.0,1.0,0.0],
         [0.0,0.0,1.0],
        ]
    ),
      start_points=numpy.array([1,1,1]),
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
    ]), space)


    res = labeling._gpu_labeling(image)
    print(res)