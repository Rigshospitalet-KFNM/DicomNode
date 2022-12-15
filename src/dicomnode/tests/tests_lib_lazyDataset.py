from unittest import TestCase, skipIf

from pydicom import Dataset
from pydicom.uid import SecondaryCaptureImageStorage
import tracemalloc

from dicomnode.lib.dicom import make_meta
from dicomnode.lib.lazyDataset import LazyDataset
import numpy


class LazyDatasetTestCase(TestCase):
  def setUp(self):
    pass

  def tearDown(self) -> None:
    pass

  def test_Laziness(self):
    ds = Dataset()

    ds.SOPClassUID = SecondaryCaptureImageStorage
    make_meta(ds)

    ds.Rows = 4096
    ds.Columns = 4096

    data = numpy.random.random_integers(0,65535, size=(4096,4096))

    ds.PixelData = data.tobytes()
