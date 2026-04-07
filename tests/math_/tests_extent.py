"""This file is just to get an easy 100% coverage"""

# Python standard library

# Third party modules

# Dicomnode modules
from dicomnode.math.extent import Extent

# Test modules
from tests.helpers.dicomnode_test_case import DicomnodeTestCase

class ExtentTestcases(DicomnodeTestCase):
  def test_extent_construction_arg_a_list_of_args(self):
    extent = Extent(5,6,7)

    self.assertEqual(extent.x, 7)
    self.assertEqual(extent.y, 6)
    self.assertEqual(extent.z, 5)

  def test_extent_cannot_default_construct(self):
    self.assertRaises(ValueError, Extent)

  def test_extent_negative_numbers_as_args(self):
    self.assertRaises(OverflowError,Extent,-1,-2,-3)

  def test_extent_false_of_different_dimensions(self):
    extent_1 = Extent(1,2,3)
    extent_2 = Extent(1,2,3,4)

    self.assertNotEqual(extent_1, extent_2)

  def test_extent_incomparable_raises_type_error(self):
    with self.assertRaises(TypeError):
      side_effect = Extent(1,2,3,4) == 4.3212

  def test_extent_really_stupid_construction(self):
    self.assertRaises(ValueError, Extent, [2,3,4], 21,4)
