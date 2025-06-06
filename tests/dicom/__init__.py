from tests.helpers.dicomnode_test_case import DicomnodeTestCase

from pydicom import Dataset, DataElement

from dicomnode import dicom

class DicomTests(DicomnodeTestCase):
  def test_add_private_tags_to_empty_dataset(self):
    test_dataset = Dataset()

    private_de = DataElement(0x1337_0101, 'IS', 1512)

    dicom.add_private_tag(test_dataset, private_de)

    self.assertIn(private_de.tag, test_dataset)


  def test_add_private_tags_group_allocation_fails(self):
    test_dataset = Dataset()

    private_de = DataElement(0x1337_0001, 'LO', "1512")

    self.assertRaises(ValueError, dicom.add_private_tag, test_dataset, private_de)

    self.assertNotIn(private_de.tag, test_dataset)

  def test_is_private_group_tag(self):
    self.assertTrue(dicom.is_private_group_tag(0x1337_0001))
    self.assertFalse(dicom.is_private_group_tag(0x1337_0100))
    self.assertFalse(dicom.is_private_group_tag(0x1336_0001))


from . import tests_nifti
from . import tests_series
from . import tests_series
