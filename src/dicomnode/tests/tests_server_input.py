from unittest import TestCase

from logging import StreamHandler
from pathlib import Path
from pydicom.uid import UID, SecondaryCaptureImageStorage
from pydicom import Dataset
from typing import List, Dict, Any, Callable, Iterator
from sys import stdout


import shutil
import logging


from dicomnode.lib.dicom import gen_uid, make_meta
from dicomnode.lib.io import load_dicom, save_dicom
from dicomnode.lib.exceptions import InvalidDataset, IncorrectlyConfigured
from dicomnode.server.input import AbstractInput


log_format = "%(asctime)s %(name)s %(levelname)s %(message)s"
correct_date_format = "%Y/%m/%d %H:%M:%S"


logging.basicConfig(
        level=logging.CRITICAL + 1 ,
        format=log_format,
        datefmt=correct_date_format,
        handlers=[StreamHandler(
          stream=stdout
        )]
      )
logger = logging.getLogger("test_server_input")

SERIES_DESCRIPTION = "Fancy Series"

class TestInput(AbstractInput):
  required_tags: List[int] = []
  required_values: Dict[int, Any] = {
    0x0008103E : SERIES_DESCRIPTION
  }

  def validate(self) -> bool:
    return True

class InputTestCase(TestCase):
  def setUp(self) -> None:
    self.path = Path(self._testMethodName)
    self.options = TestInput.Options(
      data_directory=self.path
    )
    self.test_input = TestInput(None, self.options)
    self.logger = logger

  def tearDown(self) -> None:
    shutil.rmtree(self.path)

  def test_SOPInstanceUID_is_required(self):
    self.assertIn(0x00080018, self.test_input.required_tags)

  def test_insertions(self):
    dataset = Dataset()
    self.assertRaises(InvalidDataset, self.test_input.add_image, dataset)
    dataset.SOPInstanceUID = gen_uid()
    self.assertRaises(InvalidDataset, self.test_input.add_image, dataset)
    dataset.SeriesDescription = 'Some other Description'
    self.assertRaises(InvalidDataset, self.test_input.add_image, dataset)
    dataset.SeriesDescription = SERIES_DESCRIPTION
    dataset.SOPClassUID = SecondaryCaptureImageStorage
    make_meta(dataset)
    self.test_input.add_image(dataset)
    self.assertTrue(self.test_input.getPath(dataset).exists()) # type: ignore

  def test_get_path(self):
    dataset = Dataset()
    SOPInstanceUID = gen_uid()
    dataset.SOPInstanceUID = SOPInstanceUID
    self.assertEqual(self.test_input.getPath(dataset).name, f'image_{SOPInstanceUID.name}.dcm') # type: ignore
    dataset.Modality = 'CT'
    self.assertEqual(self.test_input.getPath(dataset).name, f'CT_image_{SOPInstanceUID.name}.dcm') # type: ignore
    dataset.InstanceNumber = 431
    self.assertEqual(self.test_input.getPath(dataset).name, f'CT_image_431.dcm') # type: ignore

  def test_cleanup(self):
    dataset = Dataset()
    dataset.SOPInstanceUID = gen_uid()
    dataset.SeriesDescription = SERIES_DESCRIPTION
    dataset.SOPClassUID = SecondaryCaptureImageStorage
    make_meta(dataset)
    self.test_input.add_image(dataset)
    self.test_input._clean_up()
    self.assertFalse(self.test_input.getPath(dataset).exists())

  def test_get_data(self):
    dataset = Dataset()
    dataset.SOPInstanceUID = gen_uid()
    dataset.SeriesDescription = SERIES_DESCRIPTION
    dataset.SOPClassUID = SecondaryCaptureImageStorage
    make_meta(dataset)
    self.test_input.add_image(dataset)
    self.assertEqual(list(self.test_input.get_data()), [dataset])

  def test_load_on_creation(self):
    dataset_1 = Dataset()
    dataset_1.SOPInstanceUID = gen_uid()
    dataset_1.SeriesDescription = SERIES_DESCRIPTION
    dataset_1.SOPClassUID = SecondaryCaptureImageStorage
    make_meta(dataset_1)
    ds_1_path = self.path / "ds_1.dcm"
    save_dicom(ds_1_path, dataset_1)

    dataset_2 = Dataset()
    dataset_2.SOPInstanceUID = gen_uid()
    dataset_2.SeriesDescription = SERIES_DESCRIPTION
    dataset_2.SOPClassUID = SecondaryCaptureImageStorage
    make_meta(dataset_2)
    ds_2_path = self.path / "ds_2.dcm"
    save_dicom(ds_2_path, dataset_2)

    test_input = TestInput(None, options=TestInput.Options(data_directory=self.path))

    self.assertEqual(len(test_input),2)
    self.assertIn(dataset_1.SOPInstanceUID, test_input)
    self.assertIn(dataset_2.SOPInstanceUID, test_input)

  def test_get_path_with_in_memory_input(self):
    input = TestInput()

    dataset = Dataset()
    dataset.SOPInstanceUID = gen_uid()
    dataset.SeriesDescription = SERIES_DESCRIPTION
    dataset.SOPClassUID = SecondaryCaptureImageStorage
    make_meta(dataset)

    self.assertRaises(IncorrectlyConfigured,  input.getPath, dataset)

  def test_customer_logger(self):
    input = TestInput(options=TestInput.Options(logger=logger))

    self.assertIs(input.logger, logger)

  def test_lazy_testInput(self):
    input = TestInput(None, options=TestInput.Options(data_directory=self.path, lazy=True))

    dataset = Dataset()
    dataset.SOPInstanceUID = gen_uid()
    dataset.SeriesDescription = SERIES_DESCRIPTION
    dataset.SOPClassUID = SecondaryCaptureImageStorage
    make_meta(dataset)
    input.add_image(dataset)

    self.assertTrue(input.getPath(dataset).exists())
