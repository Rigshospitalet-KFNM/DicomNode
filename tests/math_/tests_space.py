# Python standard library

# Third party modules
import numpy
from nibabel.nifti1 import Nifti1Image, Nifti1Header

# Dicomnode modules
from dicomnode.math.space import Space

# Testing helpers
from tests.helpers.dicomnode_test_case import DicomnodeTestCase

class SpaceTestCases(DicomnodeTestCase):
  def test_construct_space_from_nifti(self):
    shape = (5,6,7)
    header = Nifti1Header()
    affine = numpy.array([
      [3.0, 0.0, 0.0, 0.0],
      [0.0, 3.0, 0.0, 0.0],
      [0.0, 0.0, 4.0, 0.0],
      [0.0, 0.0, 0.0, 1.0]
    ])
    data = numpy.random.random(shape)
    image = Nifti1Image(data, affine, header)

    space = Space.from_nifti(image)

    # To do asserts

  def test_construct_space_from_nifti_no_affine(self):
    shape = (5,6,7)
    header = Nifti1Header()
    data = numpy.random.random(shape)
    image = Nifti1Image(data, None, header)

    space = Space.from_nifti(image)

    # To do asserts


  def test_construct_space_from_nifti_f_order(self):
    shape = (5,6,7)
    header = Nifti1Header()
    data = numpy.asfortranarray(numpy.random.random(shape))
    affine = numpy.array([
      [3.0, 0.0, 0.0, 0.0],
      [0.0, 3.0, 0.0, 0.0],
      [0.0, 0.0, 4.0, 0.0],
      [0.0, 0.0, 0.0, 1.0]
    ])
    image = Nifti1Image(data, affine, header)

    space = Space.from_nifti(image)

    # To do asserts
    self.assertTrue((space.domain == numpy.array([s for s in reversed(shape)])).all())