from argparse import ArgumentTypeError
from pydicom import Dataset
from unittest import TestCase

from dicomnode.lib.dicom import gen_uid

from tests.helpers import bench
from dicomnode.lib.utils import str2bool


class Lib_util_TestCase(TestCase):
  def test_str2bool(self):
    # Trues
    self.assertTrue(str2bool(True))
    self.assertTrue(str2bool("yes"))
    self.assertTrue(str2bool("true"))
    self.assertTrue(str2bool("True"))
    self.assertTrue(str2bool("y"))
    self.assertTrue(str2bool("Y"))
    self.assertTrue(str2bool("t"))
    self.assertTrue(str2bool("1"))
    # Falses
    self.assertFalse(str2bool(False))
    self.assertFalse(str2bool("False"))
    self.assertFalse(str2bool("false"))
    self.assertFalse(str2bool("F"))
    self.assertFalse(str2bool("f"))
    self.assertFalse(str2bool("0"))
    self.assertFalse(str2bool("no"))
    self.assertFalse(str2bool("n"))
    # Raises
    self.assertRaises(ArgumentTypeError, str2bool, "maybe")
    self.assertRaises(ArgumentTypeError, str2bool, "truth")
    self.assertRaises(ArgumentTypeError, str2bool, "Alternative Facts!")
    self.assertRaises(ArgumentTypeError, str2bool, "YES!")

class pydicomTestCases(TestCase):
  @bench
  def performance_1000_datasets_method_1(self): # This is marginally Slower
    datasets = []
    for _ in range(10000):
      dataset = Dataset()
      dataset.SOPInstanceUID = gen_uid()
      dataset.SeriesInstanceUID = gen_uid()
      dataset.StudyInstanceUID = gen_uid()
      dataset.PatientID = "Helloworld"
      datasets.append(dataset)

  @bench
  def performance_1000_datasets_method_2(self): # This is marginally Faster
    datasets = []
    for _ in range(10000):
      dataset = Dataset()
      dataset.add_new(0x0020000D,'UI',gen_uid())
      dataset.add_new(0x0020000E,'UI',gen_uid())
      dataset.add_new(0x00080016,'UI',gen_uid())
      dataset.add_new(0x00100020, 'LO', "Helloworld")
      datasets.append(dataset)
