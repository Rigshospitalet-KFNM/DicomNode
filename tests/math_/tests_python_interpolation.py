from enum import Enum
from unittest import skipIf, skip

import numpy

from dicomnode.math import CUDA, switch_ordering
from dicomnode.math.image import Image
from dicomnode.math.space import Space
from dicomnode.math import _cpp

from dicomnode.math.interpolation import cpu_interpolate, resample

from tests.helpers.dicomnode_test_case import DicomnodeTestCase

class RESAMPLE_METHODS(Enum):
  LINEAR = "linear"

def old_interpolate(source: Image, target: Space, method=RESAMPLE_METHODS.LINEAR):
    from scipy.interpolate import RegularGridInterpolator
    original_grid = [numpy.arange(s) for s in reversed(source.space.extent)]

    # Create interpolator for original data
    interpolator = RegularGridInterpolator(
        tuple(original_grid),
        switch_ordering(source.raw),
        method=method.value,
        bounds_error=False,
        fill_value=source.minimum_value
    )

    # Create new grid coordinates
    new_coords = numpy.array([i for i in target.coords()])

    world_coords_new = target.starting_point + new_coords @ target.basis

    # Transform world coordinates back to original basis indices for interpolation
    orig_indices = (world_coords_new - source.space.starting_point) @ source.space.inverted_basis

    # Interpolate
    interpolated: numpy.ndarray = interpolator(orig_indices).reshape(target.extent) # type: ignore


class CPUInterpolationTests(DicomnodeTestCase):
  def test_basicInterpolation(self):
    # Assemble
    shape = (10, 20, 30)
    x = numpy.linspace(0, shape[0] - 1, shape[0])
    y = numpy.linspace(0, shape[1] - 1, shape[1])
    z = numpy.linspace(0, shape[2] - 1, shape[2])
    X, Y, Z = numpy.meshgrid(x, y, z, indexing='ij')
    data = -(((X - 10)/20)**2 + ((Y - 10)/20)**2 + ((Z - 10)/20)**2) + 0.25

    # Define original and new coordinate systems
    original_basis = numpy.array([
        [1.0, 0.0, 0.0],    # x basis vector
        [0.0, 1.0, 0.0],    # y basis vector
        [0.0, 0.0, 1.0]     # z basis vector
    ])
    original_start = numpy.array([0.0, 0.0, 0.0])

    # Define a rotated and scaled coordinate system
    theta = numpy.pi / 6  # 30 degrees rotation
    scaling = 0.25
    new_basis = numpy.array([
        [numpy.cos(theta), -numpy.sin(theta), 0.0],
        [numpy.sin(theta),  numpy.cos(theta), 0.0],
        [0.0, 0.0, 1.5]  # stretched z axis
    ]) * scaling

    new_start = numpy.array([5.0, 5.0, 5.0])
    new_shape = (32, 16, 8)

    new_space = Space(new_basis, new_start, new_shape)

    original_space = Space(original_basis, original_start, shape)
    original_image = Image(data, original_space)

    # Act
    interpolated = cpu_interpolate(
        original_image,
        new_space
    )

    # Assert
    self.assertIsInstance(interpolated.raw, numpy.ndarray)
    self.assertEqual(interpolated.raw.shape, new_shape)
    self.assertTrue(interpolated.raw.flags.c_contiguous)

  def test_incrementByInterpolation(self):
    shape = (4,4,4)
    data = (2 * numpy.arange(numpy.prod(shape), dtype=numpy.float32) + 1).reshape(shape)


    out_shape = (3,3,3)
    out_space = Space(numpy.eye(3, dtype=numpy.float32), (0.5,0,0), out_shape)

    image = Image(data, Space(numpy.eye(3, dtype=numpy.float32), (0,0,0), shape))

    out = cpu_interpolate(image, out_space)

    self.assertTrue((out.raw==(data[:3,:3,:3] + 1)).all())


class GPUInterpolationTest(DicomnodeTestCase):
  @skipIf(not CUDA, "Need GPU for this test")
  def test_that_gpu_method_works(self):
    shape = (4,4,4)
    data = (2 * numpy.arange(numpy.prod(shape), dtype=numpy.float32) + 1).reshape(shape)

    image = Image(data, Space(numpy.eye(3, dtype=numpy.float32), (0,0,0), shape))

    out_shape = (3,3,3)
    out_space = Space(numpy.eye(3, dtype=numpy.float32), (0.5,0,0), out_shape)
    dumb_image = Image(data, out_space)

    image = Image(data, Space(numpy.eye(3, dtype=numpy.float32), (0,0,0), shape))
    out = resample(image, dumb_image)

    self.assertTrue((out.raw==(data[:3,:3,:3] + 1)).all())


  @skipIf(not CUDA, "Need GPU for gpu test")
  def test_interpolation_gpu(self):
    from dicomnode.math import _cuda

    shape = (3,4,3)

    data = numpy.array([
      10.0, 30.0, 40.0,
      20.0, 50.0, 70.0,
      310.0, 130.0, 240.0,
      320.0, 150.0, 270.0,
      110.0, 130.0, 140.0,
      120.0, 150.0, 170.0,
      10.0, 30.0, 40.0,
      20.0, 50.0, 70.0,
      -10.0, 160.0, -40.0,
      -20.0, 150.0, -720.0,
      -5.0, 350.0, 40.0,
      -50.0, -150.0,-720.0,
    ], dtype=numpy.float32).reshape(shape)

    space = Space(numpy.eye(3, dtype=numpy.float32), [0,0,0], shape)

    image = Image(data, space)

    error, arr = _cuda.interpolation.linear(image, space)

    self.assertFalse(error) # Here dicomnodeError = 0 so this should be false.
    self.assertTrue((arr == data).all())

    cpu_version = cpu_interpolate(
        image,
        space
    )
    self.assertTrue((arr == cpu_version.raw).all())

  @skipIf(not CUDA, "Need GPU for gpu test")
  def test_interpolation_gpu_large(self):
    from dicomnode.math import _cuda
    # This is ~0.1s on my machine, dedicated hardware ftw
    shape = (100, 400, 400)

    data = numpy.arange(numpy.prod(shape), dtype=numpy.float32).reshape(shape)

    space = Space(numpy.eye(3, dtype=numpy.float32), [0,0,0], shape)

    image = Image(data, space)

    error, arr = _cuda.interpolation.linear(image, space)

    self.assertFalse(error) # Here dicomnodeError = 0 so this should be false.
    self.assertTrue((arr == data).all())

  @skipIf(not CUDA, "Need GPU for gpu test")
  def test_interpolationDifferentStartingPoint(self):
    from dicomnode.math import _cuda

    shape = (3, 3, 16)

    data = numpy.arange(numpy.prod(shape), dtype=numpy.float32).reshape(shape)

    space = Space(numpy.eye(3, dtype=numpy.float32), [0,0,0], shape)

    image = Image(data, space)

    out_shape = (2,2,15)

    out_space = Space(numpy.eye(3, dtype=numpy.float32), [1,1,1], out_shape)

    error, arr = _cuda.interpolation.linear(image, out_space)

    self.assertFalse(error)
    self.assertEqual(arr.shape, out_shape)
    self.assertTrue((arr==data[1:,1:,1:]).all())
    cpu_version = cpu_interpolate(
        image,
        out_space
    )
    self.assertTrue((arr == cpu_version.raw).all())

  @skipIf(not CUDA, "Need GPU for gpu test")
  def test_interpolation_middle_of_point(self):
    from dicomnode.math import _cuda

    shape = (3, 3, 6)
    # These are odd numbers
    data = (2 * numpy.arange(numpy.prod(shape), dtype=numpy.float32) + 1).reshape(shape)

    space = Space(numpy.eye(3, dtype=numpy.float32), [0,0,0], shape)
    image = Image(data, space)

    out_shape = (3,3, 5)
    out_space = Space(numpy.eye(3, dtype=numpy.float32), [0.5,0,0], out_shape)

    error, arr = _cuda.interpolation.linear(image, out_space)

    self.assertFalse(error)
    self.assertEqual(arr.shape, out_shape)
    self.assertTrue((arr==data[:,:,1:] - 1).all())
    cpu_version = cpu_interpolate(
        image,
        out_space
    )

    self.assertTrue((arr == cpu_version.raw).all())

  @skipIf(not CUDA, "Need GPU for gpu test")
  def test_interpolation_change_of_basis(self):
    from dicomnode.math import _cuda

    shape = (8, 8, 8)
    # These are odd numbers
    data = numpy.arange(numpy.prod(shape), dtype=numpy.float32).reshape(shape)

    space = Space(numpy.eye(3, dtype=numpy.float32), [0,0,0], shape)
    image = Image(data, space)

    out_shape = (4,4,4)
    out_space = Space((numpy.eye(3, dtype=numpy.float32) * 2), [0.0,0.0,0.0], out_shape)

    error, arr = _cuda.interpolation.linear(image, out_space)

    self.assertFalse(error)
    self.assertEqual(arr.shape, out_shape)
    self.assertTrue((arr==data[0::2, 0::2,0::2]).all())
    cpu_version = cpu_interpolate(
        image,
        out_space
    )
    self.assertTrue((arr == cpu_version.raw).all())

  @skipIf(not CUDA, "Need GPU for gpu test")
  def test_interpolation_change_of_both_basis(self):
    from dicomnode.math import _cuda

    shape = (8, 8, 8)
    # These are odd numbers
    data = numpy.arange(numpy.prod(shape), dtype=numpy.float32).reshape(shape)

    space = Space((2 * numpy.eye(3, dtype=numpy.float32)), [0,0,0], shape)
    image = Image(data, space)

    out_shape = (4,4,4)
    out_space = Space((numpy.eye(3, dtype=numpy.float32) / (1/2)), [0.0,0.0,0.0], out_shape)

    error, arr = _cuda.interpolation.linear(image, out_space)

    cpu_version = cpu_interpolate(
        image,
        out_space
    )

    self.assertFalse(error)
    self.assertEqual(arr.shape, out_shape)
    self.assertTrue((arr==cpu_version.raw).all())

  def test_cpp_implementation(self):
    # Assemble
    shape = (10, 20, 30)
    x = numpy.linspace(0, shape[0] - 1, shape[0])
    y = numpy.linspace(0, shape[1] - 1, shape[1])
    z = numpy.linspace(0, shape[2] - 1, shape[2])
    X, Y, Z = numpy.meshgrid(x, y, z, indexing='ij')
    data = -(((X - 10)/20)**2 + ((Y - 10)/20)**2 + ((Z - 10)/20)**2) + 0.25

    # Define original and new coordinate systems
    original_basis = numpy.array([
        [1.0, 0.0, 0.0],    # x basis vector
        [0.0, 1.0, 0.0],    # y basis vector
        [0.0, 0.0, 1.0]     # z basis vector
    ])
    original_start = numpy.array([0.0, 0.0, 0.0])

    # Define a rotated and scaled coordinate system
    theta = numpy.pi / 6  # 30 degrees rotation
    scaling = 0.25
    new_basis = numpy.array([
        [numpy.cos(theta), -numpy.sin(theta), 0.0],
        [numpy.sin(theta),  numpy.cos(theta), 0.0],
        [0.0, 0.0, 1.5]  # stretched z axis
    ]) * scaling

    new_start = numpy.array([5.0, 5.0, 5.0])
    new_shape = (32, 16, 8)

    new_space = Space(new_basis, new_start, new_shape)

    original_space = Space(original_basis, original_start, shape)
    original_image = Image(data, original_space)

    # Act
    error, interpolated = _cpp.interpolation.linear(
        original_image,
        new_space
    )

    # Assert
    self.assertIsInstance(interpolated, numpy.ndarray)
    self.assertEqual(interpolated.shape, new_shape)
    self.assertTrue(interpolated.flags.c_contiguous)

  def test_cpp_interpolation_change_of_both_basis(self):
    shape = (8, 8, 8)
    # These are odd numbers
    data = numpy.arange(numpy.prod(shape), dtype=numpy.float32).reshape(shape)

    space = Space((2 * numpy.eye(3, dtype=numpy.float32)), [0,0,0], shape)
    image = Image(data, space)

    out_shape = (4,4,4)
    out_space = Space((numpy.eye(3, dtype=numpy.float32) / (1/2)), [0.0,0.0,0.0], out_shape)

    error, arr = _cpp.interpolation.linear(image, out_space)

    cpu_version = cpu_interpolate(
        image,
        out_space
    )

    self.assertFalse(error)
    self.assertEqual(arr.shape, out_shape)

    self.assertTrue((arr==cpu_version.raw).all())



  def test_cpp_incrementByInterpolation(self):
    shape = (4,4,4)
    data = (2 * numpy.arange(numpy.prod(shape), dtype=numpy.float32) + 1).reshape(shape)


    out_shape = (3,3,3)
    out_space = Space(numpy.eye(3, dtype=numpy.float32), (0.5,0,0), out_shape)

    image = Image(data, Space(numpy.eye(3, dtype=numpy.float32), (0,0,0), shape))

    out = cpu_interpolate(image, out_space)

    self.assertTrue((out.raw==(data[:3,:3,:3] + 1)).all())

    success, stuff = _cpp.interpolation.linear(image, out_space)

    self.assertTrue((stuff == out.raw).all())