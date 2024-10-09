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
from dicomnode.lib.utils import str2bool, colomn_to_row_major_order

# Testing helpers
from tests.helpers import bench


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
      dataset.add_new(0x00100020,'LO',"Helloworld")
      datasets.append(dataset)


  def test_row_to_column(self):
    input_array = numpy.arange(4*3*2).reshape((4,3,2))
    test_array = colomn_to_row_major_order(input_array)

    self.assertEqual(test_array.shape, (2,3,4))
    to_list = [[[ elem for elem in subsublist] # Convert to build-in lists
                for subsublist in sublist]
                for sublist in test_array ]
    self.assertListEqual(to_list, [
      [[ 0.0,  6.0,  12.0, 18.0],
       [ 2.0,  8.0, 14.0, 20.0],
       [ 4.0, 10.0, 16.0, 22.0],
      ],
      [[ 1.0,  7.0, 13.0, 19.0],
       [ 3.0,  9.0, 15.0, 21.0],
       [ 5.0, 11.0, 17.0, 23.0],
      ]
    ])
