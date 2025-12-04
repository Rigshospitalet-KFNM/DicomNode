
__author__ = "Christoffer Vilstrup Jensen"

# Python Standard Library
import logging
from logging import StreamHandler
from pathlib import Path
import shutil
from sys import stdout
from typing import List, Dict, Any, Iterable, Callable
from unittest import TestCase
import datetime


# Third party Packages
from pydicom import Dataset
from pydicom.uid import SecondaryCaptureImageStorage

# Dicomnode packages
from dicomnode.constants import DICOMNODE_LOGGER_NAME
from dicomnode.dicom.series import DicomSeries
from dicomnode.dicom import gen_uid, make_meta
from dicomnode.lib.exceptions import InvalidDataset, InvalidRootDataDirectory
from dicomnode.server.grinders import Grinder, ListGrinder
from dicomnode.server.input import AbstractInput, DynamicInput
from dicomnode.server.pipeline_tree import PipelineTree, InputContainer, PatientNode

#
from tests.helpers import generate_numpy_datasets
from tests.helpers.dicomnode_test_case import DicomnodeTestCase

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
  def __call__(self, datasets: Iterable[Dataset]) -> str:
    return "GrinderString"

class TestInput1(AbstractInput):
  required_tags = []
  required_values = {
    0x0008103E : SERIES_DESCRIPTION
  }

  def validate(self) -> bool:
    return True

class TestInput2(AbstractInput):
  required_tags = [0x0010_0010]
  required_values = {
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


class PipelineTestCase(DicomnodeTestCase):
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
    super().tearDown()
    shutil.rmtree(self.path)

  def test_add_image(self):
    CPR = "1502799995"
    dataset = Dataset()
    with self.assertLogs(DICOMNODE_LOGGER_NAME) as captured_logs:
      self.assertRaises(InvalidDataset, self.pipeline_tree.add_image, dataset)
    self.assertRegexIn("0x100020 not in dataset", captured_logs.output)

    dataset.PatientID = CPR
    with self.assertLogs(DICOMNODE_LOGGER_NAME) as captured_logs:
      self.assertRaises(InvalidDataset, self.pipeline_tree.add_image, dataset)
    self.assertRegexIn("dataset was rejected from all inputs", captured_logs.output)

    dataset.SOPInstanceUID = gen_uid()
    dataset.SeriesDescription = SERIES_DESCRIPTION
    dataset.SOPClassUID = SecondaryCaptureImageStorage
    make_meta(dataset)
    with self.assertLogs(DICOMNODE_LOGGER_NAME, level=logging.DEBUG) as captured_logs:
      self.pipeline_tree.add_image(dataset)
      data = self.pipeline_tree.get_patient_input_container(CPR)

    if data is not None:
      self.assertEqual(data['arg_1'].images, 1)
      self.assertEqual(data['arg_2'], 'GrinderString')
    else:
      raise AssertionError

    self.assertRegexIn(f"Getting Patient node: {CPR}", captured_logs.output)


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
    with self.assertLogs(DICOMNODE_LOGGER_NAME, level=logging.DEBUG):
      self.pipeline_tree.clean_up_patient(CPR_1)


    self.assertEqual(self.pipeline_tree.images, 1)
    self.assertNotIn(CPR_1, self.pipeline_tree.data)
    self.assertIn(CPR_2, self.pipeline_tree.data)

    with self.assertLogs(DICOMNODE_LOGGER_NAME, level=logging.DEBUG):
      self.pipeline_tree.clean_up_patient(CPR_2)

    self.assertEqual(self.pipeline_tree.images, 0)
    self.assertNotIn(CPR_2, self.pipeline_tree.data)

  def test_remove_patients_from_pipeline_tree(self):
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

    with self.assertLogs(DICOMNODE_LOGGER_NAME, level=logging.DEBUG) as logs:
      self.pipeline_tree.clean_up_patients([CPR_1, CPR_2])

    self.assertRegexIn("Removed 2 of 3 Patients", logs.output)

    self.assertNotIn(CPR_1, self.pipeline_tree.data)
    self.assertNotIn(CPR_2, self.pipeline_tree.data)
    self.assertIn(CPR_3, self.pipeline_tree.data)
    self.assertEqual(self.pipeline_tree.images, 1)

  def test_empty_dataset_raises_exception(self):
    dataset = Dataset()
    dataset.PatientID = None
    dataset.SOPClassUID = SecondaryCaptureImageStorage
    make_meta(dataset)

    self.assertRaises(InvalidDataset, self.pipeline_tree.add_image, dataset)

  def test_pipeline_tree_to_string(self):
    class DummyInput(AbstractInput):
      def validate(self) -> bool:
        return False

    tree = PipelineTree(0x0010_0020, {
      "dummy_1" : DummyInput,
      "dummy_2" : DummyInput
    })

    tree.add_images(
      generate_numpy_datasets(10, Rows=10, Cols=10, PatientID="p1")
    )

    tree.add_images(
      generate_numpy_datasets(10, Rows=10, Cols=10, PatientID="p2")
    )

    self.assertRegexIn("Pipeline Tree - 2 Patients - 40 images total", [str(tree)])

class PatientNodeTestCase(DicomnodeTestCase):
  def setUp(self) -> None:
    self.path = Path(self._testMethodName)
    self.options = PatientNode.Options(
      container_path=self.path
    )
    self.PatientNode = PatientNode({
      'arg_1' : TestInput1,
      'arg_2' : TestInput2
    }, self.options)

  def tearDown(self) -> None:
    super().tearDown()
    shutil.rmtree(self.path)

  def test_add_image_to_patient_node(self):
    CPR = "1502799995"
    dataset = Dataset()
    # Test 1
    with self.assertLogs(DICOMNODE_LOGGER_NAME) as captured_logs:
      self.assertRaises(InvalidDataset, self.PatientNode.add_image, dataset)
    self.assertRegexIn("dataset was rejected from all inputs", captured_logs.output)
    # Test 2
    dataset.PatientID = CPR
    with self.assertLogs(DICOMNODE_LOGGER_NAME) as captured_logs:
      self.assertRaises(InvalidDataset, self.PatientNode.add_image, dataset)
    self.assertRegexIn("dataset was rejected from all inputs", captured_logs.output)
    dataset.SOPInstanceUID = gen_uid()
    dataset.SeriesDescription = SERIES_DESCRIPTION
    dataset.SOPClassUID = SecondaryCaptureImageStorage
    make_meta(dataset)
    # Test 3
    with self.assertLogs(DICOMNODE_LOGGER_NAME, level=logging.DEBUG) as captured_logs:
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
    self.assertRaises(InvalidRootDataDirectory, PatientNode,{ 'arg_1' : TestInput1, 'arg_2' : TestInput2}, options)
    path.unlink(missing_ok=True)

  def test_IC_cleanup(self):
    path = self.path / "test"
    path.mkdir()
    options = PatientNode.Options(
      container_path=path
    )
    input_container = PatientNode({ 'arg_1' : TestInput1, 'arg_2' : TestInput2}, options)
    input_container.clean_up()
    self.assertFalse(path.exists())

  def test_header_creation(self):
    options = PatientNode.Options()

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

  def test_header_creation_with_pivot_input(self):
    options = PatientNode.Options()

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

  def test_header_creation_with_dynamic_input(self):
    options = PatientNode.Options()

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

  def test_patient_node_to_string(self):
    node = PatientNode({
      'arg_1' : TestInput1,
      'arg_2' : TestInput2,
    })

    node.creation_time = datetime.datetime(2020,6,10,10,20,30,125912)

    series = DicomSeries([ds for ds in generate_numpy_datasets(10, Rows=10, Cols=10)])
    series[0x0008103E] = SERIES_DESCRIPTION

    node.add_images(series)

    self.assertEqual(str(node),
                     "PatientNode created at 2020-06-10 10:20:30.125912\n"
                     "  TestInput1 - 10 images - Valid: True\n"
                     "  TestInput2 - 0 images - Valid: True")

  def test_patient_node_with_proxy_arg(self):
    class PETInput(AbstractInput):
      required_values = {
        0x0008_0060 : "PT"
      }

      def validate(self) -> bool:
        return True

    class CTInput(AbstractInput):
      required_values = {
        0x0008_0060 : "CT"
      }

      image_grinder = ListGrinder()


      def validate(self) -> bool:
        return True

    node = PipelineTree(0x0010_0020,{
      "test" : PETInput | CTInput
    })

    series = DicomSeries([ds for ds in generate_numpy_datasets(10, Rows=10, Cols=10, PatientID="test")])
    series["Modality"] = "CT"

    added_images = node.add_images(series)
    self.assertEqual(added_images, 10)

    input_container = node.get_patient_input_container("test")
    self.assertIn("test", input_container)
    self.assertIsInstance(input_container["test"], List)
