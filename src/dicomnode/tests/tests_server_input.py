from unittest import TestCase

from logging import StreamHandler
from pathlib import Path
from pydicom.uid import UID, SecondaryCaptureImageStorage
from pydicom import Dataset
from typing import List, Dict, Any
from sys import stdout


import shutil
import logging


from dicomnode.lib.dicom import gen_uid, make_meta
from dicomnode.lib.exceptions import InvalidDataset
from dicomnode.server.input import AbstractInput


log_format = "%(asctime)s %(name)s %(levelname)s %(message)s"
correct_date_format = "%Y/%m/%d %H:%M:%S"

logging.basicConfig(
        level=logging.DEBUG,
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
    self.test_input = TestInput(instance_directory=self.path)
    self.logger = logger

  def tearDown(self) -> None:
    shutil.rmtree(self.path)

  def test_SOPInstanceUID_is_required(self):
    self.assertIn(0x00080018, self.test_input.required_tags)

  def test_Invalid_insertions(self):
    dataset = Dataset()
    self.assertRaises(InvalidDataset, self.test_input.add_image, dataset)
    dataset.SOPInstanceUID = gen_uid()
    self.assertRaises(InvalidDataset, self.test_input.add_image, dataset)
    dataset.SeriesDescription = SERIES_DESCRIPTION
    dataset.SOPClassUID = SecondaryCaptureImageStorage
    make_meta(dataset)
    print(dataset)
    self.test_input.add_image(dataset)