# Python standard library
from unittest import TestCase

# Third party packages
from pydicom import Dataset

# Dicomnode package
from dicomnode.math.space import Space

from tests.helpers.dicomnode_test_case import DicomnodeTestCase

class AffineTestCases(DicomnodeTestCase):
  def test_construct_affine(self):
    def build_dataset(i):
      ds = Dataset()

      ds.ImagePositionPatient = [-10,-10, i * 3 + -10]
      ds.ImageOrientationPatient = [1,0,0,0,1,0]
      ds.PixelSpacing = [3,3]
      ds.SliceThickness = 3
      ds.InstanceNumber = i + 1
      ds.Rows = 10
      ds.Columns = 10

      return ds

    affine = Space.from_datasets([build_dataset(i) for i in range(10)])
