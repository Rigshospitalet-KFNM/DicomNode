"""This ensures that the cpp solutions and the GPU module are producing the same
results"""

# Python standard library
from unittest import TestCase, skipIf, skip

# Third party packages
import numpy
from numpy.random import normal

# Dicomnode packages
from dicomnode import math
from dicomnode.math.types import MirrorDirection

# Test helper packages
from tests.helpers import bench
from tests.helpers.dicomnode_test_case import DicomnodeTestCase

test_iterations = 5

class MirrorTestCase(DicomnodeTestCase):

  @skipIf(not math.CUDA, "Needs Cuda compare functionality between cuda and Numpy")
  def test_mirror_x(self):
    for _ in range(test_iterations):
      data = normal(0,1, (5,5,5))
      data_gpu = data.copy()

      cpu = math.mirror(data, MirrorDirection.X)
      math.mirror_inplace_gpu(data_gpu, MirrorDirection.X)

      self.assertTrue((data_gpu == cpu).all())

  @skipIf(not math.CUDA, "Needs Cuda compare functionality between cuda and Numpy")
  def test_mirror_y(self):
    for _ in range(test_iterations):
      data = normal(0,1, (5,5,5))
      data_gpu = data.copy()

      math.mirror_inplace_gpu(data_gpu, MirrorDirection.Y)
      cpu = math.mirror(data, MirrorDirection.Y)

      self.assertTrue((data_gpu == cpu).all())

  @skipIf(not math.CUDA, "Needs Cuda compare functionality between cuda and Numpy")
  def test_mirror_z(self):
    for _ in range(test_iterations):
      data = normal(0,1, (5,5,5))
      data_gpu = data.copy()

      math.mirror_inplace_gpu(data_gpu, MirrorDirection.Z)
      cpu = math.mirror(data, MirrorDirection.Z)

      self.assertTrue((data_gpu == cpu).all())

  @skipIf(not math.CUDA, "Needs Cuda compare functionality between cuda and Numpy")
  def test_mirror_xy(self):
    for _ in range(test_iterations):
      data = normal(0,1, (5,5,5))
      data_gpu = data.copy()

      math.mirror_inplace_gpu(data_gpu, MirrorDirection.XY)
      cpu = math.mirror(data, MirrorDirection.XY)

      self.assertTrue((data_gpu == cpu).all())

  @skipIf(not math.CUDA, "Needs Cuda compare functionality between cuda and Numpy")
  def test_mirror_xz(self):
    for _ in range(test_iterations):
      data = normal(0,1, (5,5,5))
      data_gpu = data.copy()

      math.mirror_inplace_gpu(data_gpu, MirrorDirection.XZ)
      cpu = math.mirror(data, MirrorDirection.XZ)

      self.assertTrue((data_gpu == cpu).all())

  @skipIf(not math.CUDA, "Needs Cuda compare functionality between cuda and Numpy")
  def test_mirror_yz(self):
    for _ in range(test_iterations):
      data = normal(0,1, (5,5,5))
      data_gpu = data.copy()

      math.mirror_inplace_gpu(data_gpu, MirrorDirection.YZ)
      cpu = math.mirror(data, MirrorDirection.YZ)

      self.assertTrue((data_gpu == cpu).all())

  @skipIf(not math.CUDA, "Needs Cuda compare functionality between cuda and Numpy")
  def test_mirror_xyz(self):
    for _ in range(test_iterations):
      data = normal(0,1, (5,5,5))
      data_gpu = data.copy()

      math.mirror_inplace_gpu(data_gpu, MirrorDirection.XYZ)
      cpu = math.mirror(data, MirrorDirection.XYZ)

      self.assertTrue((data_gpu == cpu).all())
