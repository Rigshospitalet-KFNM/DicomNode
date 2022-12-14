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
from dicomnode.server.pipelineTree import PipelineTree, InputContainer, PatientNode


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
    self.options = PipelineTree.Options(
      data_directory=self.path
    )
    self.pipeline_tree = PipelineTree(
      0x00100020, {
        'arg_1' : TestInput1,
        'arg_2' : TestInput2
      },self.options)


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


class PatientNodeTestCase(TestCase):
  def setUp(self) -> None:
    self.path = Path(self._testMethodName)
    self.options = PatientNode.Options(
      container_path=self.path
    )
    self.PatientNode = PatientNode({
      'arg_1' : TestInput1,
      'arg_2' : TestInput2
    }, None, self.options)

  def tearDown(self) -> None:
    shutil.rmtree(self.path)

  def test_add_image(self):
    CPR = "1502799995"
    dataset = Dataset()
    # Test 1
    self.assertRaises(InvalidDataset, self.PatientNode.add_image, dataset)
    dataset.PatientID = CPR
    # Test 2
    self.assertRaises(InvalidDataset, self.PatientNode.add_image, dataset)
    dataset.SOPInstanceUID = gen_uid()
    dataset.SeriesDescription = SERIES_DESCRIPTION
    dataset.SOPClassUID = SecondaryCaptureImageStorage
    make_meta(dataset)
    # Test 3
    self.PatientNode.add_image(dataset)
    data = self.PatientNode._get_data()
    self.assertIsInstance(data, InputContainer)
    self.assertEqual(data['arg_1'].images, 1)
    self.assertEqual(data['arg_2'], 'GrinderString')

  def test_get_AI_before_instantiated(self):
    self.assertIsInstance(self.PatientNode['arg_1'], TestInput1)
    self.assertIsInstance(self.PatientNode['arg_2'], TestInput2)

  def test_raises_error_on_file_existance(self):
    path = self.path / "test"
    with path.open(mode="w") as f:
      f.write("asdf")
    options = PatientNode.Options(container_path=path)
    self.assertRaises(InvalidRootDataDirectory, PatientNode,{ 'arg_1' : TestInput1, 'arg_2' : TestInput2}, None, options)
    path.unlink(missing_ok=True)

  def test_IC_cleanup(self):
    path = self.path / "test"
    path.mkdir()
    options = PatientNode.Options(
      container_path=path
    )
    IC = PatientNode({ 'arg_1' : TestInput1, 'arg_2' : TestInput2}, None, options)
    IC._cleanup()
    self.assertFalse(path.exists())
