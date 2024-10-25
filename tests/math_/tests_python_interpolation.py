import numpy

from dicomnode.math.interpolation import py_interpolate

from tests.helpers.dicomnode_test_case import DicomnodeTestCase

class InterpolationTest(DicomnodeTestCase):
  def test_basic_interpolation(self):
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

    # Act
    interpolated = py_interpolate(
        data,
        original_basis,
        original_start,
        new_basis,
        new_start,
        new_shape
    )

    # Assert
    self.assertIsInstance(interpolated, numpy.ndarray)
    self.assertEqual(interpolated.shape, new_shape)
    self.assertTrue(interpolated.flags.c_contiguous)

    # Visualize middle slices
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    im = ax1.imshow(data[:, :, data.shape[2]//2])
    ax1.set_title('Original Data (middle z slice)')

    im = ax2.imshow(interpolated[:, :, interpolated.shape[2]//2])
    ax2.set_title('Interpolated Data (middle z slice)')

    fig.colorbar(im, ax=[ax1, ax2])

    fig.savefig("interpolation")