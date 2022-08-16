
from asyncore import write
from unittest import TestCase
from pydicom import Dataset, DataElement, Sequence
from dicomnode.lib import io

from pydicom.filebase import DicomBytesIO
from pydicom.filewriter import write_sequence, write_data_element
from pydicom.datadict import DicomDictionary, keyword_dict

class lib_io_test_case(TestCase):
  test_tag = 0x13374269
  test_tag_sq = 0x13375005
  test_string = "hello world"
  test_data_element = DataElement(test_tag, "LO", test_string)
  test_private_tag_dict = {
    test_tag : ("LO", "1", "New Private Tag", "", "NewPrivateTag"),
    test_tag_sq : ("SQ", "1", "New Private Sequence", "", "NewPrivateSequence")
  }

  def test_update_private_tags_exampel(self):
    ds = Dataset()
    try:
      ds.NewPrivateTag
      self.assertFalse(True)
    except AttributeError as E:
      pass
    io.update_private_tags(self.test_private_tag_dict)
    ds.NewPrivateTag = self.test_string
    self.assertEqual(ds[self.test_tag], self.test_data_element)

  def test_update_private_tags_check_for_tags(self):
    io.update_private_tags(self.test_private_tag_dict)
    self.assertIn(self.test_tag, DicomDictionary)
    self.assertIn("NewPrivateTag", keyword_dict)
    self.assertIn(self.test_tag_sq, DicomDictionary)
    self.assertIn("NewPrivateSequence", keyword_dict)


  def test_load_private_tags_exampel(self):
    ds = Dataset()
    ds.add_new( self.test_tag, 'UN', self.test_string.encode())
    io.load_private_tags(ds, self.test_private_tag_dict)
    self.assertEqual(ds[self.test_tag], self.test_data_element)

  def test_load_private_tags_Sequence(self):
    ds = Dataset()
    seq_ds = Dataset()
    seq_ds.add_new(self.test_tag, 'UN', self.test_string.encode())
    ds.add_new(self.test_tag_sq, 'SQ', Sequence([seq_ds]))
    io.load_private_tags(ds, self.test_private_tag_dict)
    self.assertEqual(ds[self.test_tag_sq][0][self.test_tag], self.test_data_element)

  def test_load_private_tags_unknown_sequence(self):
    ds = Dataset()
    seq_ds = Dataset()
    seq_ds.add_new(self.test_tag, 'UN', self.test_string.encode())
    seq = DataElement(self.test_tag_sq, 'SQ', Sequence([seq_ds]))
    buffer = DicomBytesIO()
    buffer._little_endian = True
    buffer._implicit_VR = True
    write_sequence(buffer, seq, ["iso8859"])
    buffer.seek(0)
    seq_bytes = buffer.read()
    ds.add_new(self.test_tag_sq, 'UN', seq_bytes)
    io.load_private_tags(ds, self.test_private_tag_dict)
    self.assertEqual(ds[self.test_tag_sq][0][self.test_tag], self.test_data_element)

  def test_load_private_tags_known_tags(self):
    ds = Dataset()
    ds.Modality = DataElement(0x00080060, 'CS', 'OT')
    io.load_private_tags(ds, self.test_private_tag_dict)
    self.assertEqual(ds.Modality, DataElement(0x00080060, 'CS', 'OT'))
