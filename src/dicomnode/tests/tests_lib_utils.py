from argparse import ArgumentTypeError
from pydicom import Dataset
from unittest import TestCase


from dicomnode.lib.utils import getTag, str2bool


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

  def test_getTag_success(self):
    ds = Dataset()
    pid = "TestID"
    ds.PatientID = pid # PatientID Tag
    getTag_PatientID = getTag(0x00100020)
    getTag_PatientName = getTag(0x00100010)
    self.assertEqual(getTag_PatientID(ds).value, pid)
    self.assertIsNone(getTag_PatientName(ds))