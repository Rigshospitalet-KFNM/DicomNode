"""This file contains test from dicomnode.lib.utils and other tests
That do not have any home.
"""

__author__ = "Christoffer Vilstrup Jensen"

# Python Standard Library
from argparse import ArgumentTypeError
from unittest import TestCase

# Thrid Party packages
import numpy
from pydicom import Dataset

# Dicomnode packages
from dicomnode.dicom import gen_uid
from dicomnode.lib.utils import str2bool

# Testing helpers
from tests.helpers import bench
from tests.helpers.dicomnode_test_case import DicomnodeTestCase

class Lib_util_TestCase(DicomnodeTestCase):
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

class pydicomTestCases(DicomnodeTestCase):
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
      dataset.add_new(0x00100020,'LO',"Helloworld")
      datasets.append(dataset)
