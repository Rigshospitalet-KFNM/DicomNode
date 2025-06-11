# Python Standard library

# Third party Library
import numpy
from pydicom import Dataset, DataElement

# Dicomnode Modules
from dicomnode.math.space import Space
from dicomnode import dicom

# Testing helpers
from tests.helpers.dicomnode_test_case import DicomnodeTestCase

class DicomTests(DicomnodeTestCase):
  def test_add_private_tag_errors_on_public_tags(self):
    test_dataset = Dataset()
    private_de = DataElement(0x0010_0010, 'PN', "HELLO WOLRD")
    self.assertRaises(ValueError, dicom.add_private_tag, test_dataset, private_de)

    self.assertEqual(len(test_dataset), 0)

  def test_add_private_tag_errors_on_private_group_allocation(self):
    test_dataset = Dataset()
    private_de = DataElement(0x1337_0001, 'LO', "DICOMNODE")
    self.assertRaises(ValueError, dicom.add_private_tag, test_dataset, private_de)
    self.assertEqual(len(test_dataset), 0)

  def test_add_private_tag_errors_on_reserved_tags(self):
    test_dataset = Dataset()
    private_de = DataElement(0x1337_01FE, 'LO', "DICOMNODE")
    self.assertRaises(ValueError, dicom.add_private_tag, test_dataset, private_de)
    self.assertEqual(len(test_dataset), 0)
    private_de = DataElement(0x1337_01FF, 'LO', "DICOMNODE")
    self.assertRaises(ValueError, dicom.add_private_tag, test_dataset, private_de)
    self.assertEqual(len(test_dataset), 0)

  def test_add_private_tags_to_empty_dataset(self):
    test_dataset = Dataset()

    private_de = DataElement(0x1337_0101, 'IS', 1512)

    dicom.add_private_tag(test_dataset, private_de)

    self.assertIn(private_de.tag, test_dataset)

    self.assertIn(0x1337_0001, test_dataset)
    self.assertIn(0x1337_01FE, test_dataset)
    self.assertIn(0x1337_01FF, test_dataset)

  def test_add_private_tags_adding_multiple_things(self):
    test_dataset = Dataset()

    private_de = DataElement(0x1337_0101, 'IS', 1512)
    private_de = DataElement(0x1337_0102, 'IS', 1512)
    private_de = DataElement(0x1337_0103, 'IS', 1512)

    dicom.add_private_tag(test_dataset, private_de)

    self.assertIn(private_de.tag, test_dataset)

    self.assertIn(0x1337_0001, test_dataset)
    self.assertIn(0x1337_01FE, test_dataset)
    self.assertIn(0x1337_01FF, test_dataset)



  def test_is_private_group_tag(self):
    self.assertTrue(dicom.is_private_group_tag(0x1337_0001))
    self.assertFalse(dicom.is_private_group_tag(0x1337_0100))
    self.assertFalse(dicom.is_private_group_tag(0x1336_0001))

  def test_creating_coordinate_system_with_space(self):
    testing_space = Space(
      [[6,0,0], [0,7,0], [0,0,8]], #type: ignore
      [0,0,0],
      [2,3,4]
    )

    points, (x_dim, y_dim, z_dim), orientation, starting_point = dicom.create_dicom_coordinate_system(testing_space)

    self.assertEqual(x_dim, 6)
    self.assertEqual(y_dim, 7)
    self.assertEqual(z_dim, 8)

    self.assertListEqual(orientation, [1,0,0,0,1,0])
    self.assertListEqual([p for p in starting_point], [0,0,0])

from . import tests_nifti
from . import tests_series
from . import tests_series
