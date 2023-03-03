from unittest import TestCase

from pydicom import Dataset

from pydicom.uid import ExplicitVRBigEndian, ExplicitVRLittleEndian, ImplicitVRLittleEndian, CTImageStorage
from dicomnode.lib.dicom import getTag, make_meta, gen_uid
from dicomnode.lib.exceptions import InvalidDataset

class DicomTestCase(TestCase):
  def test_getTag_success(self):
    ds = Dataset()
    pid = "TestID"
    ds.PatientID = pid # PatientID Tag
    getTag_PatientID = getTag(0x00100020)
    getTag_PatientName = getTag(0x00100010)
    self.assertEqual(getTag_PatientID(ds).value, pid)
    self.assertIsNone(getTag_PatientName(ds))

  def test_make_meta_missing_SOPClassUID(self):
    ds = Dataset()
    self.assertRaises(InvalidDataset, make_meta, ds)

  def test_make_meta_ImplicitVRLittleEndian(self):
    ds = Dataset()
    ds.SOPClassUID = CTImageStorage
    ds.is_implicit_VR = True
    ds.is_little_endian = True
    make_meta(ds)
    self.assertEqual(ds.file_meta.TransferSyntaxUID, ImplicitVRLittleEndian)

  def test_make_meta_ExplicitVRLittleEndian(self):
    ds = Dataset()
    ds.SOPClassUID = CTImageStorage
    ds.is_implicit_VR = False
    ds.is_little_endian = True
    make_meta(ds)
    self.assertEqual(ds.file_meta.TransferSyntaxUID, ExplicitVRLittleEndian)

  def test_make_meta_ExplicitVRBigEndian(self):
    ds = Dataset()
    ds.SOPClassUID = CTImageStorage
    ds.is_implicit_VR = False
    ds.is_little_endian = False
    make_meta(ds)
    self.assertEqual(ds.file_meta.TransferSyntaxUID, ExplicitVRBigEndian)

  def test_make_meta_ImplicitVRBigEndian(self):
    ds = Dataset()
    ds.SOPClassUID = CTImageStorage
    ds.is_implicit_VR = True
    ds.is_little_endian = False
    self.assertRaises(InvalidDataset, make_meta, ds)
