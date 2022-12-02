from unittest import TestCase, skipIf

from pydicom import Dataset
import tracemalloc
from dicomnode.lib.lazyDataset import LazyDataset
try:
  import numpy
  NUMPY_IMPORTED = True
except ImportError:
  NUMPY_IMPORTED = False


class LazyDatasetTestCase(TestCase):
  @skipIf(not NUMPY_IMPORTED, "Numpy Required for test")
  def test_Laziness(self):
    ds = Dataset()
