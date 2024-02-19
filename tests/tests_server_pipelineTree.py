
__author__ = "Christoffer Vilstrup Jensen"

# Python Standard Library
import logging
from logging import StreamHandler
from pathlib import Path
import shutil
from sys import stdout
from typing import List, Dict, Any, Iterator, Callable
from unittest import TestCase
import datetime


# Third party Packages
from pydicom import Dataset
from pydicom.uid import SecondaryCaptureImageStorage

# Dicomnode packages
from dicomnode.lib.dicom import gen_uid, make_meta
from dicomnode.lib.exceptions import InvalidDataset, InvalidRootDataDirectory
from dicomnode.lib.dicom_factory import Blueprint, StaticElement, InstanceCopyElement, CopyElement
from dicomnode.lib.numpy_factory import NumpyFactory
from dicomnode.server.grinders import Grinder
from dicomnode.server.input import AbstractInput, DynamicInput
from dicomnode.server.pipeline_tree import PipelineTree, InputContainer, PatientNode

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

class TestGrinder(Grinder):
  def __call__(self, datasets: Iterator[Dataset]) -> str:
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

  image_grinder: Grinder = TestGrinder()

  def validate(self) -> bool:
    return True


class TestDynamicInput(DynamicInput):
  required_tags = [0x00100010]
  required_values = {}

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
      }, self.options)


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
    data = self.pipeline_tree.get_patient_input_container(CPR)
    if data is not None:
      self.assertEqual(data['arg_1'].images, 1)
      self.assertEqual(data['arg_2'], 'GrinderString')
    else:
      raise AssertionError

  def test_remove_expired_studies(self):
    CPR_1 = "1502799995"
    dataset_1 = Dataset()
    dataset_1.PatientID = CPR_1
    dataset_1.SOPClassUID = SecondaryCaptureImageStorage
    dataset_1.SeriesDescription = SERIES_DESCRIPTION
    dataset_1.SOPInstanceUID = gen_uid()
    make_meta(dataset_1)
    CPR_2 = "1210131111"
    dataset_2 = Dataset()
    dataset_2.PatientID = CPR_2
    dataset_2.SOPClassUID = SecondaryCaptureImageStorage
    dataset_2.SeriesDescription = SERIES_DESCRIPTION
    dataset_2.SOPInstanceUID = gen_uid()
    make_meta(dataset_2)

    self.pipeline_tree.add_image(dataset_1)
    self.pipeline_tree.add_image(dataset_2)

    patient_node_1 = self.pipeline_tree[CPR_1]
    patient_node_2 = self.pipeline_tree[CPR_2]

    self.assertIsInstance(patient_node_1, PatientNode)
    self.assertIsInstance(patient_node_2, PatientNode)

    # Monkey patching creation time circumventing, injection of datetime.
    patient_node_1.creation_time = datetime.datetime(2002, 5, 7) # type: ignore
    patient_node_2.creation_time = datetime.datetime(2002, 6, 7) # type: ignore

    expiry_time = datetime.datetime(2002,5,15)

    self.pipeline_tree.remove_expired_studies(expiry_time)

    self.assertNotIn(CPR_1, self.pipeline_tree)
    self.assertIn(CPR_2, self.pipeline_tree)

  def test_remove_patient(self):
    CPR_1 = "1502799995"
    dataset_1 = Dataset()
    dataset_1.PatientID = CPR_1
    dataset_1.SOPClassUID = SecondaryCaptureImageStorage
    dataset_1.SeriesDescription = SERIES_DESCRIPTION
    dataset_1.SOPInstanceUID = gen_uid()
    make_meta(dataset_1)
    CPR_2 = "1210131111"
    dataset_2 = Dataset()
    dataset_2.PatientID = CPR_2
    dataset_2.SOPClassUID = SecondaryCaptureImageStorage
    dataset_2.SeriesDescription = SERIES_DESCRIPTION
    dataset_2.SOPInstanceUID = gen_uid()
    make_meta(dataset_2)

    self.pipeline_tree.add_image(dataset_1)
    self.pipeline_tree.add_image(dataset_2)

    self.assertEqual(self.pipeline_tree.images, 2)
    self.pipeline_tree.clean_up_patient(CPR_1)
    self.assertEqual(self.pipeline_tree.images, 1)
    self.assertNotIn(CPR_1, self.pipeline_tree.data)
    self.assertIn(CPR_2, self.pipeline_tree.data)
    self.pipeline_tree.clean_up_patient(CPR_2)
    self.assertEqual(self.pipeline_tree.images, 0)
    self.assertNotIn(CPR_2, self.pipeline_tree.data)

  def test_remove_patients(self):
    CPR_1 = "1502799995"
    dataset_1 = Dataset()
    dataset_1.PatientID = CPR_1
    dataset_1.SOPClassUID = SecondaryCaptureImageStorage
    dataset_1.SeriesDescription = SERIES_DESCRIPTION
    dataset_1.SOPInstanceUID = gen_uid()
    make_meta(dataset_1)
    CPR_2 = "1210131111"
    dataset_2 = Dataset()
    dataset_2.PatientID = CPR_2
    dataset_2.SOPClassUID = SecondaryCaptureImageStorage
    dataset_2.SeriesDescription = SERIES_DESCRIPTION
    dataset_2.SOPInstanceUID = gen_uid()
    make_meta(dataset_2)
    CPR_3 = "1111550641"
    dataset_3 = Dataset()
    dataset_3.PatientID = CPR_3
    dataset_3.SOPClassUID = SecondaryCaptureImageStorage
    dataset_3.SeriesDescription = SERIES_DESCRIPTION
    dataset_3.SOPInstanceUID = gen_uid()
    make_meta(dataset_3)

    self.pipeline_tree.add_image(dataset_1)
    self.pipeline_tree.add_image(dataset_2)
    self.pipeline_tree.add_image(dataset_3)

    self.pipeline_tree.clean_up_patients([CPR_1, CPR_2])

    self.assertNotIn(CPR_1, self.pipeline_tree.data)
    self.assertNotIn(CPR_2, self.pipeline_tree.data)
    self.assertIn(CPR_3, self.pipeline_tree.data)
    self.assertEqual(self.pipeline_tree.images, 1)

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
    data = self.PatientNode.extract_input_container()
    self.assertIsInstance(data, InputContainer)
    self.assertEqual(data['arg_1'].images, 1)
    self.assertEqual(data['arg_2'], 'GrinderString')

  def test_get_AI_before_instantiated(self):
    self.assertIsInstance(self.PatientNode['arg_1'], TestInput1)
    self.assertIsInstance(self.PatientNode['arg_2'], TestInput2)

  def test_raises_error_on_file_existence(self):
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
    input_container = PatientNode({ 'arg_1' : TestInput1, 'arg_2' : TestInput2}, None, options)
    input_container.clean_up()
    self.assertFalse(path.exists())

  def test_header_creation(self):
    blueprint = Blueprint([
      StaticElement(0x00100010, 'PN', "Anon^Mus"),
      CopyElement(0x00100020),
    ])

    options = PatientNode.Options(
      header_blueprint=blueprint,
      factory=NumpyFactory(),
    )

    patient_node = PatientNode({
      'arg_1' : TestInput1
    }, options=options)


    CPR = "1502799995"
    dataset = Dataset()
    dataset.PatientID = CPR
    dataset.SOPInstanceUID = gen_uid()
    dataset.SeriesDescription = SERIES_DESCRIPTION
    dataset.SOPClassUID = SecondaryCaptureImageStorage
    make_meta(dataset)

    patient_node.add_image(dataset)

    input_container = patient_node.extract_input_container()

    self.assertIsNotNone(input_container.header)

  def test_header_creation_with_pivot_input(self):
    blueprint = Blueprint([
      StaticElement(0x00100010, 'PN', "Anon^Mus"),
      CopyElement(0x00100020),
    ])

    options = PatientNode.Options(
      header_blueprint=blueprint,
      factory=NumpyFactory(),
      parent_input="arg_1"
    )

    patient_node = PatientNode({
      'arg_1' : TestInput1
    }, options=options)


    CPR = "1502799995"
    dataset = Dataset()
    dataset.PatientID = CPR
    dataset.SOPInstanceUID = gen_uid()
    dataset.SeriesDescription = SERIES_DESCRIPTION
    dataset.SOPClassUID = SecondaryCaptureImageStorage
    make_meta(dataset)

    patient_node.add_image(dataset)

    input_container = patient_node.extract_input_container()

    self.assertIsNotNone(input_container.header)

  def test_header_creation_with_dynamic_input(self):
    blueprint = Blueprint([
      StaticElement(0x00100010, 'PN', "Anon^Mus"),
      CopyElement(0x00100020),
    ])

    options = PatientNode.Options(
      header_blueprint=blueprint,
      factory=NumpyFactory(),
      parent_input="arg_1"
    )

    patient_dynamic_node = PatientNode({
      'arg_1' : TestDynamicInput
    }, options=options)

    CPR = "1502799995"
    dataset_1 = Dataset()
    dataset_1.PatientID = CPR
    dataset_1.PatientName = "Foo Bar"
    dataset_1.SOPInstanceUID = gen_uid()
    dataset_1.SeriesInstanceUID = gen_uid()
    dataset_1.SeriesDescription = SERIES_DESCRIPTION
    dataset_1.SOPClassUID = SecondaryCaptureImageStorage
    make_meta(dataset_1)
    patient_dynamic_node.add_image(dataset_1)

    dataset_2 = Dataset()
    dataset_2.PatientID = CPR
    dataset_2.PatientName = "Foo Bar"
    dataset_2.SOPInstanceUID = gen_uid()
    dataset_2.SeriesInstanceUID = gen_uid()
    dataset_2.SeriesDescription = SERIES_DESCRIPTION
    dataset_2.SOPClassUID = SecondaryCaptureImageStorage
    make_meta(dataset_2)
    patient_dynamic_node.add_image(dataset_2)

    input_container = patient_dynamic_node.extract_input_container()
