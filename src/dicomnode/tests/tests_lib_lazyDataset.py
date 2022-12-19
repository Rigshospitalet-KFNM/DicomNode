import gc as garbage
from pathlib import Path
import shutil
from sys import getrefcount
import tracemalloc
from unittest import TestCase, skipIf


import numpy
from pydicom import Dataset
from pydicom.uid import SecondaryCaptureImageStorage

from dicomnode.lib.dicom import make_meta
from dicomnode.lib.lazyDataset import LazyDataset
from dicomnode.lib.io import save_dicom, load_dicom

from dicomnode.tests.helpers import generate_numpy_datasets

class LazyDatasetTestCase(TestCase):
  def setUp(self):
    self.path = Path(f"{self._testMethodName}")
    self.path.mkdir()

  def tearDown(self) -> None:
    shutil.rmtree(self.path, ignore_errors=False)

  def test_Laziness(self):
    tracemalloc.start(100)
    garbage.collect()
    before_size, before_peak = tracemalloc.get_traced_memory()
    tracemalloc.reset_peak()

    ds = list(generate_numpy_datasets(1, Cols=4096, Rows=4096, Bits=32, rescale=False))[0]
    cpr = "1502799995"
    ds.PatientID = cpr
    ds.PatientName = "Bla^Blur^Blum"

    ds.SOPClassUID = SecondaryCaptureImageStorage
    make_meta(ds)
    target_Path = self.path / "image.dcm"

    save_dicom(target_Path,ds)
    ds_size, ds_peak = tracemalloc.get_traced_memory()

    del ds

    garbage.collect()
    tracemalloc.reset_peak()

    lazy_ds = LazyDataset(target_Path)

    lazy_size, Lazy_peak = tracemalloc.get_traced_memory()

    self.assertEqual(lazy_ds.PatientID, cpr)

    realized_size, realized_peak = tracemalloc.get_traced_memory()

    ds_size = ds_size - before_size
    lazy_size = lazy_size - before_size
    realized_size = realized_size - before_size

    self.assertLess(lazy_size, ds_size / 100) # Divide by 100 to ensure that it's not random noise, that causes this test to pass
    self.assertLess(lazy_size, realized_size / 100)
    tracemalloc.stop()

  def test_set_first(self):
    ds = list(generate_numpy_datasets(1, Cols=40, Rows=40, Bits=32, rescale=False))[0]
    cpr = "1502799995"
    ds.PatientID = cpr
    ds.PatientName = "Bla^Blur^Blum"

    ds.SOPClassUID = SecondaryCaptureImageStorage
    make_meta(ds)
    target_Path = self.path / "image.dcm"

    save_dicom(target_Path,ds)

    lazy_ds = LazyDataset(target_Path)
    lazy_ds.PatientSex = 'M'

    self.assertEqual(lazy_ds.PatientID, cpr)
    self.assertEqual(lazy_ds.PatientSex, "M")
    self.assertEqual(lazy_ds.SOPClassUID, SecondaryCaptureImageStorage)

  def test_del_first(self):
    ds = list(generate_numpy_datasets(1, Cols=40, Rows=40, Bits=32, rescale=False))[0]
    cpr = "1502799995"
    ds.PatientID = cpr
    ds.PatientName = "Bla^Blur^Blum"

    ds.SOPClassUID = SecondaryCaptureImageStorage
    make_meta(ds)
    target_Path = self.path / "image.dcm"

    save_dicom(target_Path,ds)

    lazy_ds = LazyDataset(target_Path)
    del lazy_ds.PatientID

    self.assertNotIn(0x00100020,lazy_ds)

  def test_del_wrapped(self):
    ds = list(generate_numpy_datasets(1, Cols=40, Rows=40, Bits=32, rescale=False))[0]
    cpr = "1502799995"
    ds.PatientID = cpr
    ds.PatientName = "Bla^Blur^Blum"
    ds.SOPClassUID = SecondaryCaptureImageStorage
    make_meta(ds)
    target_Path = self.path / "image.dcm"

    save_dicom(target_Path,ds)
    lazy_ds = LazyDataset(target_Path)
    self.assertRaises(TypeError, lazy_ds.__delattr__, "_wrapped")

  def test_isDataset(self):
    ds = list(generate_numpy_datasets(1, Cols=40, Rows=40, Bits=32, rescale=False))[0]
    cpr = "1502799995"
    ds.PatientID = cpr
    ds.PatientName = "Bla^Blur^Blum"

    ds.SOPClassUID = SecondaryCaptureImageStorage
    make_meta(ds)
    target_Path = self.path / "image.dcm"

    save_dicom(target_Path,ds)

    lazy_ds = LazyDataset(target_Path)
    self.assertIsInstance(lazy_ds, Dataset)
