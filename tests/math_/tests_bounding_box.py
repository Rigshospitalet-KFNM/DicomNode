# Python standard library
from random import randint
from time import time
from unittest import TestCase, skipIf

# Third party modules
import numpy


# Dicomnode modules
from dicomnode.math import CUDA, _bounding_box_cpu, _bounding_box_gpu, bounding_box

# Testing modules


# region Tests
class BoundingBoxTestCase(TestCase):
  def test_line(self):
    array = numpy.zeros((10))

    array[3] = 1
    array[5] = 1
    array[7] = 1

    bbox = _bounding_box_cpu(array)

    self.assertEqual(len(bbox),1)
    self.assertEqual(len(bbox[0]), 2)
    self.assertEqual(bbox[0][0], 3)
    self.assertEqual(bbox[0][1], 7)

  def test_plane(self):
    array = numpy.zeros((10, 10))

    array[7][3] = 1
    array[5][5] = 1
    array[3][7] = 1

    bbox = _bounding_box_cpu(array)

    self.assertEqual(len(bbox),2)
    self.assertEqual(len(bbox[0]), 2)
    self.assertEqual(len(bbox[1]), 2)
    self.assertEqual(bbox[0][0], 3)
    self.assertEqual(bbox[0][1], 7)
    self.assertEqual(bbox[1][0], 3)
    self.assertEqual(bbox[1][1], 7)

  def test_plane_2(self):
    array = numpy.zeros((12, 10))

    array[7][3] = 1
    array[7][5] = 1
    array[7][7] = 1

    bbox = _bounding_box_cpu(array)

    self.assertEqual(len(bbox),2)
    self.assertEqual(len(bbox[0]), 2)
    self.assertEqual(len(bbox[1]), 2)
    self.assertEqual(bbox[0][0], 3)
    self.assertEqual(bbox[0][1], 7)
    self.assertEqual(bbox[1][0], 7)
    self.assertEqual(bbox[1][1], 7)

  def test_plane_3(self):
    array = numpy.zeros((12, 10))

    array[3][3] = 1
    array[5][5] = 1
    array[7][7] = 1

    bbox = _bounding_box_cpu(array)

    self.assertEqual(len(bbox),2)
    self.assertEqual(len(bbox[0]), 2)
    self.assertEqual(len(bbox[1]), 2)
    self.assertEqual(bbox[0][0], 3)
    self.assertEqual(bbox[0][1], 7)
    self.assertEqual(bbox[1][0], 3)
    self.assertEqual(bbox[1][1], 7)

  def test_cube(self):
    array = numpy.zeros((12, 10, 4))

    array[3][3][3] = 1
    array[5][5][3] = 1
    array[7][7][3] = 1

    (x_min, x_max), (y_min, y_max), (z_min, z_max) = bounding_box(array)

    self.assertEqual(x_min, 3)
    self.assertEqual(x_max, 3)
    self.assertEqual(y_min, 3)
    self.assertEqual(y_max, 7)
    self.assertEqual(z_min, 3)
    self.assertEqual(z_max, 7)

  def test_randomized(self):
    for _ in range(100):
      array = numpy.zeros((11, 11, 11))
      x_min = randint(0, 5)
      x_max = randint(5, 10)
      y_min = randint(0, 5)
      y_max = randint(5, 10)
      z_min = randint(0, 5)
      z_max = randint(5, 10)

      array[z_min][y_min][x_min] = 1
      array[z_max][y_max][x_max] = 1

      (bx_min, bx_max), (by_min, by_max), (bz_min, bz_max) = bounding_box(array)
      self.assertEqual(x_min, bx_min)
      self.assertEqual(x_max, bx_max)
      self.assertEqual(y_min, by_min)
      self.assertEqual(y_max, by_max)
      self.assertEqual(z_min, bz_min)
      self.assertEqual(z_max, bz_max)

  def performance_big_box(self):
    array = numpy.zeros((650, 440, 440))
    x_min = randint(0, 220)
    x_max = randint(220, 439)
    y_min = randint(0, 220)
    y_max = randint(220, 439)
    z_min = randint(0, 325)
    z_max = randint(325, 649)

    array[z_min][y_min][x_min] = 1
    array[z_max][y_max][x_max] = 1
    (bx_min, bx_max), (by_min, by_max), (bz_min, bz_max) = bounding_box(array)

    self.assertEqual(x_min, bx_min)
    self.assertEqual(x_max, bx_max)
    self.assertEqual(y_min, by_min)
    self.assertEqual(y_max, by_max)
    self.assertEqual(z_min, bz_min)
    self.assertEqual(z_max, bz_max)

  @skipIf(not CUDA, "Need a GPU to do bounding box")
  def test_gpu_big_box(self):
    array = numpy.zeros((650, 440, 440))
    x_min = randint(0, 220)
    x_max = randint(220, 439)
    y_min = randint(0, 220)
    y_max = randint(220, 439)
    z_min = randint(0, 325)
    z_max = randint(325, 649)

    array[z_min][y_min][x_min] = 1
    array[z_max][y_max][x_max] = 1
    (bx_min, bx_max), (by_min, by_max), (bz_min, bz_max) = _bounding_box_gpu(array)

    self.assertEqual(x_min, bx_min)
    self.assertEqual(x_max, bx_max)
    self.assertEqual(y_min, by_min)
    self.assertEqual(y_max, by_max)
    self.assertEqual(z_min, bz_min)
    self.assertEqual(z_max, bz_max)

  def performance_other_solution(self):
    array = numpy.zeros((650, 440, 440))
    x_min = randint(0, 220)
    x_max = randint(220, 439)
    y_min = randint(0, 220)
    y_max = randint(220, 439)
    z_min = randint(0, 325)
    z_max = randint(325, 649)

    array[z_min][y_min][x_min] = 1
    array[z_max][y_max][x_max] = 1

    def bbox(mask):
      """ Returns a bounding box from binary image

      FSL Approach src/avwutils/fslstats.cc (Line: 243-279)

      input:
        3D binary

      output:
        xmin xsize ymin ysize zmin zsize

      """
      # x_idx = np.where(np.any(mask, axis=0))[0]
      # x1, x2 = x_idx[[0, -1]]

      # y_idx = np.where(np.any(mask, axis=1))[0]
      # y1, y2 = y_idx[[0, -1]]

      # return np.array([y1,x1,y2,x2])

      imadim = mask.shape

      xmin = imadim[0]-1
      xmax = 0
      ymin = imadim[1]-1
      ymax = 0
      zmin = imadim[2]-1
      zmax = 0

      for z in range(0,imadim[2]):
          for y in range(0,imadim[1]):
              for x in range(0,imadim[0]):
                  if mask[x,y,z]:
                      if x<xmin : xmin=x
                      if x>xmax : xmax=x
                      if y<ymin : ymin=y
                      if y>ymax : ymax=y
                      if z<zmin : zmin=z
                      if z>zmax : zmax=z

      return xmin,1+xmax-xmin,ymin,1+ymax-ymin,zmin,1+zmax-zmin

    start = time()
    bx_min, bx_max, by_min, by_max, bz_min, bz_max = bbox(array)
    stop = time()
    print("RH kinetics", stop - start)

    print(x_min, bx_min)
    print(x_max, bx_max)
    print(y_min, by_min)
    print(y_max, by_max)
    print(z_min, bz_min)
    print(z_max, bz_max)
