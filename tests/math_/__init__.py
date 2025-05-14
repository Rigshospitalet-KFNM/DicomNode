from . import tests_affine
from . import tests_bounding_box
from . import tests_mirror
from . import tests_image
from . import tests_types
from . import tests_space

import numpy
from dicomnode.math import switch_ordering

from tests.helpers.dicomnode_test_case import DicomnodeTestCase

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

  def test_switch_ordering_is_it(self):
    pass