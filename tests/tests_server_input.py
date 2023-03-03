from unittest import TestCase
from logging import StreamHandler
import numpy
from pathlib import Path
from pydicom.uid import UID, SecondaryCaptureImageStorage
from pydicom import Dataset
from typing import List, Dict, Any, Callable, Iterator
from sys import stdout

import shutil
import logging

from tests.helpers import generate_numpy_datasets

from dicomnode.lib.dimse import Address
from dicomnode.lib.dicom import gen_uid, make_meta
from dicomnode.lib.dicomFactory import Blueprint
from dicomnode.lib.numpyFactory import NumpyFactory
from dicomnode.lib.grinders import numpy_grinder
from dicomnode.lib.io import load_dicom, save_dicom
from dicomnode.lib.exceptions import InvalidDataset, IncorrectlyConfigured
from dicomnode.server.input import AbstractInput, HistoricAbstractInput, DynamicInput

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

class TestDynamicInput(DynamicInput):
  required_tags = [0x0020000D, 0x00100020, 0x00080018]
  image_grinder = staticmethod(numpy_grinder)

  def validate(self) -> bool:
    return len(self.data) >= 2

# Note the functional tests of historic inputs can be found in tests_server_nodes.py
class FaultyHistoricInput(HistoricAbstractInput):
  required_tags: List[int] = []
  required_values: Dict[int, Any] = {
    0x0008103E : SERIES_DESCRIPTION
  }

  def validate(self) -> bool:
    return True

class FaultyBlueprintHistoricInput(HistoricAbstractInput):
  required_tags: List[int] = []
  required_values: Dict[int, Any] = {
    0x0008103E : SERIES_DESCRIPTION
  }

  c_move_blueprint = Blueprint([])

  def validate(self) -> bool:
    return True

class HistoricInput(HistoricAbstractInput):
  required_tags: List[int] = []
  required_values: Dict[int, Any] = {
    0x0008103E : SERIES_DESCRIPTION
  }

  address = Address('localhost', 50001, "DUMMY")
  c_move_blueprint = Blueprint([])

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
    self.assertTrue(self.test_input.get_path(dataset).exists()) # type: ignore

  def test_get_path(self):
    dataset = Dataset()
    SOPInstanceUID = gen_uid()
    dataset.SOPInstanceUID = SOPInstanceUID
    self.assertEqual(self.test_input.get_path(dataset).name, f'image_{SOPInstanceUID.name}.dcm') # type: ignore
    dataset.Modality = 'CT'
    self.assertEqual(self.test_input.get_path(dataset).name, f'CT_image_{SOPInstanceUID.name}.dcm') # type: ignore
    dataset.InstanceNumber = 431
    self.assertEqual(self.test_input.get_path(dataset).name, f'CT_image_431.dcm') # type: ignore

  def test_cleanup(self):
    dataset = Dataset()
    dataset.SOPInstanceUID = gen_uid()
    dataset.SeriesDescription = SERIES_DESCRIPTION
    dataset.SOPClassUID = SecondaryCaptureImageStorage
    make_meta(dataset)
    self.test_input.add_image(dataset)
    self.test_input._clean_up()
    self.assertFalse(self.test_input.get_path(dataset).exists())

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

    self.assertRaises(IncorrectlyConfigured,  input.get_path, dataset)

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

    self.assertTrue(input.get_path(dataset).exists())

  def test_lazy_testInput_NoPath(self):
    input = TestInput(None, options=TestInput.Options(data_directory=None, lazy=True))

    dataset = Dataset()
    dataset.SOPInstanceUID = gen_uid()
    dataset.SeriesDescription = SERIES_DESCRIPTION
    dataset.SOPClassUID = SecondaryCaptureImageStorage
    make_meta(dataset)
    self.assertRaises(IncorrectlyConfigured, input.add_image, dataset)

  def test_historic_input_missing_pivot(self):
    with self.assertLogs("dicomnode", logging.CRITICAL) as cm:
      self.assertRaises(IncorrectlyConfigured, FaultyHistoricInput)
    self.assertIn("CRITICAL:dicomnode:You forgot to parse the pivot to The Input", cm.output)

  def test_historic_input_missing_blueprint(self):
    with self.assertLogs("dicomnode", logging.CRITICAL) as cm:
      self.assertRaises(IncorrectlyConfigured, FaultyHistoricInput, Dataset())
    self.assertIn("CRITICAL:dicomnode:A C move blueprint is missing", cm.output)

  def test_historic_input_missing_address(self):
    with self.assertLogs("dicomnode", logging.CRITICAL) as cm:
      self.assertRaises(IncorrectlyConfigured, FaultyBlueprintHistoricInput, Dataset())
    self.assertIn("CRITICAL:dicomnode:A target address is needed to send a C-Move to", cm.output)

  def test_historic_input_missing_factory(self):
    with self.assertLogs("dicomnode", logging.CRITICAL) as cm:
      self.assertRaises(IncorrectlyConfigured, HistoricInput, Dataset())
    self.assertIn("CRITICAL:dicomnode:A Factory is needed to generate a C move message", cm.output)

  def test_historic_input_missing_ae_title(self):
    with self.assertLogs("dicomnode", logging.CRITICAL) as cm:
      self.assertRaises(IncorrectlyConfigured, HistoricInput, Dataset(), HistoricInput.Options(factory=NumpyFactory()))
    self.assertIn("CRITICAL:dicomnode:Historic Inputs needs a AE Title of the SCU", cm.output)

  def test_dynamic_output(self):
    patient_ID = "2002112161"
    studyUID = gen_uid()
    seriesUID_1 = gen_uid()
    seriesUID_2 = gen_uid()
    seriesUID_3 = gen_uid()
    datasets_1 = generate_numpy_datasets(2, StudyUID=studyUID, SeriesUID=seriesUID_1, Cols=10, Rows=10, PatientID = patient_ID)
    datasets_2 = generate_numpy_datasets(2, StudyUID=studyUID, SeriesUID=seriesUID_2, Cols=10, Rows=10, PatientID = patient_ID)
    datasets_3 = generate_numpy_datasets(2, StudyUID=studyUID, SeriesUID=seriesUID_3, Cols=10, Rows=10, PatientID = patient_ID)

    TDI = TestDynamicInput()

    for dataset_1, dataset_2, dataset_3 in zip(datasets_1,datasets_2,datasets_3):
      TDI.add_image(dataset_1)
      TDI.add_image(dataset_2)
      TDI.add_image(dataset_3)

    self.assertEqual(len(TDI.data),3)


  def test_dynamic_output_with_path(self):
    patient_ID = "2002112161"
    studyUID = gen_uid()
    seriesUID_1 = gen_uid()
    seriesUID_2 = gen_uid()
    seriesUID_3 = gen_uid()
    datasets_1 = generate_numpy_datasets(2, StudyUID=studyUID, SeriesUID=seriesUID_1, Cols=10, Rows=10, PatientID = patient_ID)
    datasets_2 = generate_numpy_datasets(2, StudyUID=studyUID, SeriesUID=seriesUID_2, Cols=10, Rows=10, PatientID = patient_ID)
    datasets_3 = generate_numpy_datasets(2, StudyUID=studyUID, SeriesUID=seriesUID_3, Cols=10, Rows=10, PatientID = patient_ID)

    options = TestDynamicInput.Options(data_directory=self.path)

    TDI = TestDynamicInput(options=options)

    for dataset_1, dataset_2, dataset_3 in zip(datasets_1,datasets_2,datasets_3):
      TDI.add_image(dataset_1)
      TDI.add_image(dataset_2)
      TDI.add_image(dataset_3)

    self.assertEqual(len(TDI.data),3)


  def test_dynamic_output_with_lazy(self):
    patient_ID = "2002112161"
    studyUID = gen_uid()
    seriesUID_1 = gen_uid()
    seriesUID_2 = gen_uid()
    seriesUID_3 = gen_uid()
    datasets_1 = generate_numpy_datasets(2, StudyUID=studyUID, SeriesUID=seriesUID_1, Cols=10, Rows=10, PatientID = patient_ID)
    datasets_2 = generate_numpy_datasets(2, StudyUID=studyUID, SeriesUID=seriesUID_2, Cols=10, Rows=10, PatientID = patient_ID)
    datasets_3 = generate_numpy_datasets(2, StudyUID=studyUID, SeriesUID=seriesUID_3, Cols=10, Rows=10, PatientID = patient_ID)

    options = TestDynamicInput.Options(data_directory=self.path, lazy=True)

    TDI = TestDynamicInput(options=options)

    for dataset_1, dataset_2, dataset_3 in zip(datasets_1,datasets_2,datasets_3):
      TDI.add_image(dataset_1)
      TDI.add_image(dataset_2)
      TDI.add_image(dataset_3)

    self.assertEqual(len(TDI.data),3)

  def test_dynamic_invalid_dataset(self):
    empty_dataset = Dataset()
    TDI = TestDynamicInput()
    self.assertRaises(InvalidDataset, TDI.add_image, empty_dataset)

  def test_dynamic_missing_separator_tag(self):
    dataset = Dataset()
    dataset.SOPInstanceUID = gen_uid()
    dataset.StudyInstanceUID = gen_uid()
    dataset.PatientID = "2002112161"
    TDI = TestDynamicInput()
    self.assertRaises(InvalidDataset, TDI.add_image, dataset)

  def test_silly_dynamic_input(self):
    class TestSillyDynamicInput(DynamicInput):
      required_tags = [0x0020000D, 0x00100020, 0x00080018]
      separator_tag = 0x00200013

      def validate(self) -> bool:
        return False

    dataset = Dataset()
    dataset.SOPInstanceUID = gen_uid()
    dataset.StudyInstanceUID = gen_uid()
    dataset.PatientID = "2002112161"
    dataset.InstanceNumber = 3
    TDI = TestSillyDynamicInput()
    TDI.add_image(dataset)

  def test_dynamic_input_missing_static_method(self):
    class TestSillyDynamicInput(DynamicInput):
      required_tags = [0x0020000D, 0x00100020, 0x00080018]
      image_grinder = numpy_grinder

      def validate(self) -> bool:
        return False

    dataset = Dataset()
    dataset.SOPInstanceUID = gen_uid()
    dataset.StudyInstanceUID = gen_uid()
    dataset.SeriesInstanceUID = gen_uid()
    dataset.PatientID = "2002112161"
    dataset.InstanceNumber = 3
    TDI = TestSillyDynamicInput()
    TDI.add_image(dataset)
    with self.assertLogs("dicomnode", logging.CRITICAL) as cm:
      self.assertRaises(IncorrectlyConfigured, TDI.get_data)
    self.assertIn('CRITICAL:dicomnode:image grinder is not a static method!\nFor DynamicInputs set image_grinder=staticmethod(grinder_function)', cm.output)

  def test_dynamic_get_data(self):
    patient_ID = "2002112161"
    studyUID = gen_uid()
    seriesUID_1 = gen_uid()
    seriesUID_2 = gen_uid()
    seriesUID_3 = gen_uid()
    series_images = 4
    datasets_1 = generate_numpy_datasets(series_images, StudyUID=studyUID, SeriesUID=seriesUID_1, Cols=10, Rows=10, PatientID = patient_ID)
    datasets_2 = generate_numpy_datasets(series_images, StudyUID=studyUID, SeriesUID=seriesUID_2, Cols=10, Rows=10, PatientID = patient_ID)
    datasets_3 = generate_numpy_datasets(series_images, StudyUID=studyUID, SeriesUID=seriesUID_3, Cols=10, Rows=10, PatientID = patient_ID)


    TDI = TestDynamicInput()

    for dataset_1, dataset_2, dataset_3 in zip(datasets_1,datasets_2,datasets_3):
      TDI.add_image(dataset_1)
      TDI.add_image(dataset_2)
      TDI.add_image(dataset_3)

    self.assertEqual(len(TDI.data[seriesUID_1.name]),series_images)
    self.assertEqual(len(TDI.data[seriesUID_2.name]),series_images)
    self.assertEqual(len(TDI.data[seriesUID_3.name]),series_images)

    numpyDict = TDI.get_data()

    self.assertIsInstance(numpyDict[seriesUID_1.name], numpy.ndarray)
    self.assertIsInstance(numpyDict[seriesUID_2.name], numpy.ndarray)
    self.assertIsInstance(numpyDict[seriesUID_3.name], numpy.ndarray)

    self.assertEqual(numpyDict[seriesUID_1.name].shape,(series_images,10,10))
    self.assertEqual(numpyDict[seriesUID_2.name].shape,(series_images,10,10))
    self.assertEqual(numpyDict[seriesUID_3.name].shape,(series_images,10,10))


