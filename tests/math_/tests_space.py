# Python standard library

# Third party modules
import numpy
from nibabel.nifti1 import Nifti1Image, Nifti1Header

# Dicomnode modules
from dicomnode.math.types import MirrorDirection
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
    self.assertTrue((space.extent == numpy.array([s for s in reversed(shape)])).all())

  def test_coordinates(self):
    shape = (2,3,4)

    space = Space(numpy.eye(3, dtype=numpy.float32),[0,0,0], shape)

    coords = numpy.array([i for i in space.coords()])

    self.assertTrue((coords[0] ==  [0,0,0]).all())
    self.assertTrue((coords[1] ==  [1,0,0]).all())
    self.assertTrue((coords[2] ==  [2,0,0]).all())
    self.assertTrue((coords[3] ==  [3,0,0]).all())
    self.assertTrue((coords[4] ==  [0,1,0]).all())
    self.assertTrue((coords[5] ==  [1,1,0]).all())
    self.assertTrue((coords[6] ==  [2,1,0]).all())
    self.assertTrue((coords[7] ==  [3,1,0]).all())
    self.assertTrue((coords[8] ==  [0,2,0]).all())
    self.assertTrue((coords[9] ==  [1,2,0]).all())
    self.assertTrue((coords[10] == [2,2,0]).all())
    self.assertTrue((coords[11] == [3,2,0]).all())
    self.assertTrue((coords[12] == [0,0,1]).all())
    self.assertTrue((coords[13] == [1,0,1]).all())
    self.assertTrue((coords[14] == [2,0,1]).all())
    self.assertTrue((coords[15] == [3,0,1]).all())
    self.assertTrue((coords[16] == [0,1,1]).all())
    self.assertTrue((coords[17] == [1,1,1]).all())
    self.assertTrue((coords[18] == [2,1,1]).all())
    self.assertTrue((coords[19] == [3,1,1]).all())
    self.assertTrue((coords[20] == [0,2,1]).all())
    self.assertTrue((coords[21] == [1,2,1]).all())
    self.assertTrue((coords[22] == [2,2,1]).all())
    self.assertTrue((coords[23] == [3,2,1]).all())

  def test_mirroring_of_space(self):
    space = Space(numpy.array([ # type: ignore
      [2,1,0],
      [1,2,0],
      [0,0,3]
    ]), [10,20,30], [10,30,50])

    space.mirror_perspective(MirrorDirection.X)

    print(space.basis)
    print(space.starting_point)

    space.mirror_perspective(MirrorDirection.X)

    print(space.basis)
    print(space.starting_point)
