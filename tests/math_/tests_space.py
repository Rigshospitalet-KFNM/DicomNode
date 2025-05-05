# Python standard library

# Third party modules
import numpy
from nibabel.nifti1 import Nifti1Image, Nifti1Header

# Dicomnode modules
from dicomnode.lib.exceptions import NonReducedBasis
from dicomnode.math.types import MirrorDirection
from dicomnode.math.space import Space
from dicomnode.math.types import ROTATION_MATRIX_90_DEG_X,\
  ROTATION_MATRIX_90_DEG_Y,ROTATION_MATRIX_90_DEG_Z

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

  def test_equal_with_nonsense_raises_type_error(self):
    space = Space(numpy.array([ # type: ignore
      [1,0,0],
      [0,1,0],
      [0,0,1],
    ]), [10,20,30], [10,30,50])

    with self.assertRaises(TypeError):
      space == 1 # type: ignore # The point is to trigger an error

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
    # We need some asserts here
    space = Space(numpy.array([ # type: ignore
      [2,1,0],
      [1,2,0],
      [0,0,3]
    ]), [10,20,30], [10,30,50])

    space.mirror_perspective(MirrorDirection.X)

    #print(space.basis)
    #print(space.starting_point)

    space.mirror_perspective(MirrorDirection.X)

    #print(space.basis)
    #print(space.starting_point)

  def test_rotating_a_space_that_doesnt_need_rotating(self):
    space = Space(numpy.array([ # type: ignore
      [1,0,0],
      [0,1,0],
      [0,0,1],
    ]), [10,20,30], [10,30,50])

    self.assertIsNotNone(space.reference_space)
    self.assertTrue((space.get_rotation_matrix_to_standard_space() == numpy.eye(3)).all())

# Note that the rotation testing is incomplete because it's doesn't check starting point / extent
  def test_rotation_1(self):
    space = Space(numpy.array([ # type: ignore
      [0,1,0],
      [1,0,0],
      [0,0,1]
    ]), [10,20,30], [10,30,50])

    self.assertIsNone(space.reference_space)

    new_space = space.rotate(space.get_rotation_matrix_to_standard_space())

    self.assertIsNotNone(new_space.reference_space)

  def test_rotation_2(self):
    space = Space(numpy.array([ # type: ignore
      [0,0,1],
      [0,1,0],
      [1,0,0],
    ]), [10,20,30], [10,30,50])

    self.assertIsNone(space.reference_space)

    new_space = space.rotate(space.get_rotation_matrix_to_standard_space())

    self.assertIsNotNone(new_space.reference_space)

  def test_rotation_3(self):
    space = Space(numpy.array([ # type: ignore
      [0,0,1],
      [1,0,0],
      [0,1,0],
    ]), [10,20,30], [10,30,50])

    self.assertIsNone(space.reference_space)

    new_space = space.rotate(space.get_rotation_matrix_to_standard_space())

    self.assertIsNotNone(new_space.reference_space)

  def test_rotation_4(self):
    space = Space(numpy.array([ # type: ignore
      [0,1,0],
      [0,0,1],
      [1,0,0],
    ]), [10,20,30], [10,30,50])

    self.assertIsNone(space.reference_space)

    new_space = space.rotate(space.get_rotation_matrix_to_standard_space())

    self.assertIsNotNone(new_space.reference_space)

  def test_rotation_5(self):
    space = Space(numpy.array([ # type: ignore
      [1,0,0],
      [0,0,1],
      [0,1,0],
    ]), [10,20,30], [10,30,50])

    self.assertIsNone(space.reference_space)

    new_space = space.rotate(space.get_rotation_matrix_to_standard_space())

    self.assertIsNotNone(new_space.reference_space)

  def test_really_really_really_dumb_rotation_test(self):
    space = Space(numpy.array([ # type: ignore
      [1,0,0],
      [1,1,0],
      [0,0,1],
    ]), [10,20,30], [10,30,50])

    self.assertRaises(NonReducedBasis, space.get_rotation_matrix_to_standard_space)

  def test_rotations_updates_extent_X(self):
    space = Space(numpy.array([ # type: ignore
      [0,1,0],
      [1,0,0],
      [0,0,1]
    ]), [10,20,30], [10,30,50])

    self.assertListEqual(list(space._rotate_extent(ROTATION_MATRIX_90_DEG_X)), [10,50,30])

  def test_rotations_updates_extent_Y(self):
    space = Space(numpy.array([ # type: ignore
      [0,1,0],
      [1,0,0],
      [0,0,1]
    ]), [10,20,30], [10,30,50])

    self.assertListEqual(list(space._rotate_extent(ROTATION_MATRIX_90_DEG_Y)), [50,30,10])

  def test_rotations_updates_extent_Z(self):
    space = Space(numpy.array([ # type: ignore
      [0,1,0],
      [1,0,0],
      [0,0,1]
    ]), [10,20,30], [10,30,50])

    self.assertListEqual(list(space._rotate_extent(ROTATION_MATRIX_90_DEG_Z)), [30,10,50])

  def test_rotations_updates_extent_XY(self):
    space = Space(numpy.array([ # type: ignore
      [0,1,0],
      [1,0,0],
      [0,0,1]
    ]), [10,20,30], [10,30,50])

    self.assertListEqual(list(space._rotate_extent(ROTATION_MATRIX_90_DEG_X @ ROTATION_MATRIX_90_DEG_Y)), [50,10,30])

  def test_rotations_updates_extent_YX(self):
    space = Space(numpy.array([ # type: ignore
      [0,1,0],
      [1,0,0],
      [0,0,1]
    ]), [10,20,30], [10,30,50])

    self.assertListEqual(list(space._rotate_extent(ROTATION_MATRIX_90_DEG_Y @ ROTATION_MATRIX_90_DEG_X)), [30,50,10])

  def test_starting_point_x_axis(self):
    space = Space(numpy.array([ # type: ignore
      [1,0,0],
      [0,2,0],
      [0,0,3]
    ]), [100,200,300], [20,30,40])

    space_x = space.rotate(ROTATION_MATRIX_90_DEG_X)
    space_xx = space_x.rotate(ROTATION_MATRIX_90_DEG_X)
    space_xxx = space_xx.rotate(ROTATION_MATRIX_90_DEG_X)
    rotated_space = space_xxx.rotate(ROTATION_MATRIX_90_DEG_X)
    self.assertEqual(space, rotated_space)

    inverse_x_rotation = ROTATION_MATRIX_90_DEG_X @ ROTATION_MATRIX_90_DEG_X @ ROTATION_MATRIX_90_DEG_X
    inv_x_space = space.rotate(inverse_x_rotation)
    inv_xx_space = inv_x_space.rotate(inverse_x_rotation)
    inv_xxx_space = inv_xx_space.rotate(inverse_x_rotation)

    inv_rotated_space = inv_xxx_space.rotate(inverse_x_rotation)

    self.assertEqual(inv_rotated_space, space)
    self.assertEqual(inv_x_space, space_xxx)
    self.assertEqual(inv_xx_space, space_xx)
    self.assertEqual(inv_xxx_space, space_x)

  def test_starting_point_y_axis(self):
    space = Space(numpy.array([ # type: ignore
      [1,0,0],
      [0,2,0],
      [0,0,3]
    ]), [100,200,300], [20,30,40])

    space_y = space.rotate(ROTATION_MATRIX_90_DEG_Y)
    space_yy = space_y.rotate(ROTATION_MATRIX_90_DEG_Y)
    space_yyy = space_yy.rotate(ROTATION_MATRIX_90_DEG_Y)
    rotated_space = space_yyy.rotate(ROTATION_MATRIX_90_DEG_Y)

    self.assertEqual(space, rotated_space)

    inverse_y_rotation = ROTATION_MATRIX_90_DEG_Y @ ROTATION_MATRIX_90_DEG_Y @ ROTATION_MATRIX_90_DEG_Y
    inv_y_space = space.rotate(inverse_y_rotation)
    inv_yy_space = inv_y_space.rotate(inverse_y_rotation)
    inv_yyy_space = inv_yy_space.rotate(inverse_y_rotation)

    inv_rotated_space = inv_yyy_space.rotate(inverse_y_rotation)

    self.assertEqual(inv_rotated_space, space)
    self.assertEqual(inv_y_space, space_yyy)
    self.assertEqual(inv_yy_space, space_yy)
    self.assertEqual(inv_yyy_space, space_y)


  def test_starting_point_z_axis(self):

    space = Space(numpy.array([ # type: ignore
      [1,0,0],
      [0,2,0],
      [0,0,3]
    ]), [100,200,300], [20,30,40])

    space_z = space.rotate(ROTATION_MATRIX_90_DEG_Z)
    space_zz = space_z.rotate(ROTATION_MATRIX_90_DEG_Z)
    space_zzz = space_zz.rotate(ROTATION_MATRIX_90_DEG_Z)
    rotated_space = space_zzz.rotate(ROTATION_MATRIX_90_DEG_Z)
    self.assertListEqual(list(space_z.starting_point), [120,200,300])
    self.assertListEqual(list(space_zz.starting_point), [120,260,300])
    self.assertListEqual(list(space_zzz.starting_point), [100,260,300])
    self.assertEqual(space, rotated_space)

    inverse_z_rotation = ROTATION_MATRIX_90_DEG_Z @ ROTATION_MATRIX_90_DEG_Z @ ROTATION_MATRIX_90_DEG_Z
    inv_z_space = space.rotate(inverse_z_rotation)
    inv_zz_space = inv_z_space.rotate(inverse_z_rotation)
    inv_zzz_space = inv_zz_space.rotate(inverse_z_rotation)

    inv_rotated_space = inv_zzz_space.rotate(inverse_z_rotation)

    self.assertEqual(inv_rotated_space, space)
    self.assertEqual(inv_z_space, space_zzz)
    self.assertEqual(inv_zz_space, space_zz)
    self.assertEqual(inv_zzz_space, space_z)
