import logging
from logging import StreamHandler
from pathlib import Path
from pydicom import Dataset
from pydicom.uid import SecondaryCaptureImageStorage
import shutil
from sys import stdout
from typing import List, Dict, Any, Iterator, Callable
from unittest import TestCase

from dicomnode.lib.dicom import gen_uid, make_meta
from dicomnode.lib.exceptions import InvalidDataset, InvalidRootDataDirectory
from dicomnode.server.input import AbstractInput
from dicomnode.server.pipelineTree import PipelineTree, InputContainer


SERIES_DESCRIPTION = "Fancy Description"

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

logger = logging.getLogger("test_server_pipeline")

def test_grinder(datasets: Iterator[Dataset]) -> str:
  return "GrinderString"

class TestInput1(AbstractInput):
  required_tags: List[int] = []
  required_values: Dict[int, Any] = {
    0x0008103E : SERIES_DESCRIPTION
  }

  def validate(self) -> bool:
    return True

class TestInput2(AbstractInput):
  required_tags: List[int] = [0x00100010]
  required_values: Dict[int, Any] = {
    0x0008103E : SERIES_DESCRIPTION
  }

  image_grinder: Callable[[Iterator[Dataset]], Any] = test_grinder

  def validate(self) -> bool:
    return True


class PipelineTestCase(TestCase):
  def setUp(self) -> None:
    self.path = Path(self._testMethodName)
    self.pipeline_tree = PipelineTree(
      0x00100020, {
        'arg_1' : TestInput1,
        'arg_2' : TestInput2
      }, root_data_directory=self.path)


  def tearDown(self) -> None:
    shutil.rmtree(self.path)

  def test_add_image(self):
    CPR = "1502799995"
    dataset = Dataset()
    self.assertRaises(InvalidDataset, self.pipeline_tree.add_image, dataset)
    dataset.PatientID = CPR
    self.assertRaises(InvalidDataset, self.pipeline_tree.add_image, dataset)
    dataset.SOPInstanceUID = gen_uid()
    dataset.SeriesDescription = SERIES_DESCRIPTION
    dataset.SOPClassUID = SecondaryCaptureImageStorage
    make_meta(dataset)
    self.pipeline_tree.add_image(dataset)
    data = self.pipeline_tree.validate_patient_ID(CPR)
    if data is not None:
      self.assertEqual(data['arg_1'].images, 1)
      self.assertEqual(data['arg_2'], 'GrinderString')
    else:
      raise AssertionError


class InputContainerTestCase(TestCase):
  def setUp(self) -> None:
    self.path = Path(self._testMethodName)
    self.input_container = InputContainer({
      'arg_1' : TestInput1,
      'arg_2' : TestInput2
    }, self.path)

  def tearDown(self) -> None:
    shutil.rmtree(self.path)

  def test_add_image(self):
    CPR = "1502799995"
    dataset = Dataset()
    self.assertRaises(InvalidDataset, self.input_container.add_image, dataset)
    dataset.PatientID = CPR
    self.assertRaises(InvalidDataset, self.input_container.add_image, dataset)
    dataset.SOPInstanceUID = gen_uid()
    dataset.SeriesDescription = SERIES_DESCRIPTION
    dataset.SOPClassUID = SecondaryCaptureImageStorage
    make_meta(dataset)
    self.input_container.add_image(dataset)
    data = self.input_container._get_data()
    self.assertEqual(id(data), id(self.input_container))
    self.assertEqual(data['arg_1'].images, 1)
    self.assertEqual(data['arg_2'], 'GrinderString')

  def test_get_AI_before_instantiated(self):
    self.assertIsInstance(self.input_container['arg_1'], TestInput1)
    self.assertIsInstance(self.input_container['arg_2'], TestInput2)

  def test_raises_error_on_file_existance(self):
    path = self.path / "test"
    with path.open(mode="w") as f:
      f.write("asdf")
    self.assertRaises(InvalidRootDataDirectory, InputContainer,{ 'arg_1' : TestInput1, 'arg_2' : TestInput2}, path)

  def test_IC_cleanup(self):
    path = self.path / "test"
    path.mkdir()
    IC = InputContainer({ 'arg_1' : TestInput1, 'arg_2' : TestInput2}, path)
    IC._cleanup()
    self.assertFalse(path.exists())

