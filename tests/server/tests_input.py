"""Test file for src/dicomnode/server/input.py"""

__author__ = "Christoffer Vilstrup Jensen"

# Python Standard Library
from datetime import date
import logging
from logging import StreamHandler
import os
from pathlib import Path
from typing import Any, List, Dict, Optional
import shutil
from sys import stdout
from unittest import TestCase, mock
from time import sleep

#Third Party libs

import numpy
from pydicom import Dataset
from pydicom.uid import UID, SecondaryCaptureImageStorage


# Dicomnode packages
from tests.helpers import generate_numpy_datasets, TESTING_TEMPORARY_DIRECTORY
from tests.helpers.dicomnode_test_case import DicomnodeTestCase
from dicomnode.lib.validators import RegexValidator, CaselessRegexValidator, NegatedValidator
from dicomnode.data_structures.optional import OptionalPath
from dicomnode.dicom.dimse import Address, create_query_dataset, QueryLevels
from dicomnode.dicom import gen_uid, make_meta
from dicomnode.dicom.series import DicomSeries
from dicomnode.lib.io import save_dicom, Directory
from dicomnode.lib.exceptions import InvalidDataset, IncorrectlyConfigured
from dicomnode.server.grinders import NumpyGrinder
from dicomnode.config import DicomnodeConfig, DicomnodeConfigRaw, config_from_raw
from dicomnode.server.input import AbstractInput, HistoricAbstractInput,\
  DynamicInput, AbstractInputProxy

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
  required_tags = []
  required_values: Dict[int|str, Any] = {
    0x0008103E : SERIES_DESCRIPTION
  }

  def validate(self) -> bool:
    return True



# Note the functional tests of historic inputs can be found in tests_server_nodes.py

class InputTestCase(DicomnodeTestCase):
  def setUp(self) -> None:
    os.chdir(TESTING_TEMPORARY_DIRECTORY)
    self.input_directory = Directory(Path(self._testMethodName))
    self.node_path = OptionalPath(self.input_directory.path)
    self.options = config_from_raw(DicomnodeConfigRaw(
      ARCHIVE_DIRECTORY=self._testMethodName
    ))
    self.test_input = TestInput(self.options, node_path=self.node_path)
    self.logger = logger

  def tearDown(self) -> None:
    super().tearDown()
    shutil.rmtree(self.input_directory.path)

  def test_abstract_input_infinite_job_security(self):
    a_mock = mock.Mock()

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

      def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        a_mock()

      def validate(self) -> bool:
        return True

    test = PETInput | CTInput

    self.assertIsInstance(test, type)
    self.assertTrue(issubclass(test, AbstractInputProxy))

    test_instance = test()
    self.assertEqual(test_instance.images, 0)

    self.assertIsInstance(test_instance, AbstractInputProxy)
    self.assertIsInstance(test_instance, AbstractInput)
    self.assertFalse(test_instance.validate())

    test_dataset = Dataset()
    test_dataset.SOPInstanceUID = gen_uid()
    test_dataset.SOPClassUID = SecondaryCaptureImageStorage
    test_dataset.Modality = 'CT'
    make_meta(test_dataset)

    a_mock.assert_not_called()
    added_images = test_instance.add_image(test_dataset)
    a_mock.assert_called_once()

    self.assertEqual(added_images, 1)
    # HAHAHAHAHAHAHAHAHAHAHAHAHAHA
    self.assertIsInstance(test_instance, CTInput)
    self.assertEqual(test_instance.images, 1)
    self.assertEqual(test_instance.enforce_single_series, False)


    test_dataset_2 = Dataset()
    test_dataset_2.SOPInstanceUID = gen_uid()
    test_dataset_2.SOPClassUID = SecondaryCaptureImageStorage
    test_dataset_2.Modality = 'CT'
    make_meta(test_dataset_2)

    self.assertEqual(test_instance.add_image(test_dataset_2), 1)
    a_mock.assert_called_once()

  def test_abstract_input_infinite_job_security_operations_2_electric_bogalo(self):
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

      def validate(self) -> bool:
        return True

    class MRInput(AbstractInput):
      required_values = {
        0x0008_0060 : "CT"
      }

      def validate(self) -> bool:
        return True

    class RTInput(AbstractInput):
      required_values = {
        0x0008_0060 : "RT"
      }

      def validate(self) -> bool:
        return True

    proxy_class = PETInput | CTInput | MRInput | RTInput
    self.assertEqual(proxy_class.type_options, [PETInput, CTInput, MRInput, RTInput])
    proxy_class_2 = (PETInput | CTInput) | (MRInput | RTInput)
    self.assertEqual(proxy_class_2.type_options, [PETInput, CTInput, MRInput, RTInput])
    self.assertIsNot(proxy_class, proxy_class_2)
    proxy_class_3 = PETInput | (CTInput | (MRInput | RTInput))
    self.assertEqual(proxy_class_3.type_options, [PETInput, CTInput, MRInput, RTInput])

  def test_infinite_job_security_pokemon_evolving(self):
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

      def validate(self) -> bool:
        return True

    HeHeHeHe = PETInput | CTInput

    dataset_PET = Dataset()
    dataset_PET.SOPInstanceUID = gen_uid()
    dataset_PET.Modality = 'PET'

    dataset_CT = Dataset()
    dataset_CT.SOPInstanceUID = gen_uid()
    dataset_CT.Modality = 'CT'

    HiHiHiHi = HeHeHeHe()

    with self.assertRaises(InvalidDataset):
      HiHiHiHi.add_image(dataset_PET)
      HiHiHiHi.add_image(dataset_CT)

  def test_abstract_input_replacing_images_preserves_class_invariant(self):
    class NeverValidating(AbstractInput):
      def validate(self) -> bool:
        return False

    pipeline_input = NeverValidating()

    dataset = Dataset()
    dataset.SOPInstanceUID = gen_uid()

    added_images_1 = pipeline_input.add_image(dataset)
    self.assertEqual(added_images_1, 1)
    self.assertEqual(pipeline_input.images, 1)
    added_images_2 = pipeline_input.add_image(dataset)
    self.assertEqual(added_images_2, 1)
    self.assertEqual(pipeline_input.images, 1)

  def test_enforcing_single_series(self):
    class NormalBehavior(AbstractInput):
      def validate(self) -> bool:
        return False

    class EnforcingBehavior(AbstractInput):
      enforce_single_series = True

      def validate(self) -> bool:
        return False

    dataset_missing_series = Dataset()
    dataset_missing_series.SOPInstanceUID = gen_uid()

    dataset_series_1 = Dataset()
    dataset_series_1.SOPInstanceUID = gen_uid()
    dataset_series_1.SeriesInstanceUID = gen_uid()

    dataset_series_2 = Dataset()
    dataset_series_2.SOPInstanceUID = gen_uid()
    dataset_series_2.SeriesInstanceUID = gen_uid()

    normal = NormalBehavior()

    normal.add_image(dataset_missing_series)
    normal.add_image(dataset_series_1)
    normal.add_image(dataset_series_2)

    enforcer = EnforcingBehavior()

    self.assertRaises(InvalidDataset, enforcer.add_image, dataset_missing_series)
    enforcer.add_image(dataset_series_1)
    self.assertRaises(InvalidDataset, enforcer.add_image, dataset_series_2)

  def test_validating_with_strings(self):
    dataset = Dataset()
    dataset.SOPInstanceUID = gen_uid() # This is defaulted needed by all inputs

    class StringTagsValidation(AbstractInput):
      required_tags = ["not a tag"]

    self.assertRaises(IncorrectlyConfigured, StringTagsValidation.validate_image, dataset)


    class StringValuesValidation(AbstractInput):
      required_values = {
        "Not a tag" : 0
      }

    self.assertRaises(IncorrectlyConfigured, StringValuesValidation.validate_image, dataset)

    dataset.Modality = "PT"
    dataset.InstanceNumber = 1

    class StringTagsValidationWorking(AbstractInput):
      required_tags = ["Modality"]
      required_values = {
        "InstanceNumber" : 1
      }

    self.assertTrue(StringTagsValidationWorking.validate_image(dataset))

  def test_validators(self):
    topogram = Dataset()
    topogram.SOPInstanceUID = gen_uid()
    topogram.SeriesDescription = "TOPOGRAM 1 tm"

    ct_image = Dataset()
    ct_image.SOPInstanceUID = gen_uid()
    ct_image.SeriesDescription = "AC_CT"

    class ValidatorInput(AbstractInput):
      required_values = {
        0x0008_103E : NegatedValidator(CaselessRegexValidator("topogram"))
      }

    self.assertFalse(ValidatorInput.validate_image(topogram))
    self.assertTrue(ValidatorInput.validate_image(ct_image))

  def test_abstract_input_initialization_yells_at_you(self):
    class BadProxy(AbstractInputProxy):
      type_options = []

    class WorseProxy(AbstractInputProxy):
      type_options = [int]

    class EnforcingInput(AbstractInput):
      enforce_single_study_date = True

    class NotEnforcingInput(AbstractInput):
      enforce_single_study_date = False

    self.assertRaises(IncorrectlyConfigured, BadProxy, config_from_raw())
    self.assertRaises(IncorrectlyConfigured, WorseProxy, config_from_raw())
    self.assertRaises(IncorrectlyConfigured, EnforcingInput | NotEnforcingInput, config_from_raw())


class DynamicTests(DicomnodeTestCase):
  def test_dynamic_inputs_datasets_are_stored_in_different_folders(self):
    def make_dataset(num):
      dataset = Dataset()
      dataset.SOPInstanceUID = gen_uid()
      dataset.SOPClassUID = SecondaryCaptureImageStorage
      dataset.SeriesNumber = num
      dataset.InstanceNumber = 1
      dataset.SeriesDescription = "Overlapping"
      make_meta(dataset)

      return dataset

    class TestDynamicInput(DynamicInput):
      separator_tag = 0x0020_0011

      required_tags = [0x0020_0011, 0x0008_0018]
      image_grinder = NumpyGrinder()

      def validate(self) -> bool:
        return len(self) >= 2

    config = config_from_raw(DicomnodeConfigRaw(ARCHIVE_DIRECTORY=self._testMethodName))

    input_ = TestDynamicInput(config, OptionalPath(self._testMethodName))

    series = 5

    datasets = [ make_dataset(num + 1) for num in range(series)]

    for ds in datasets:
      input_.add_image(ds)

    path = Path(self._testMethodName)

    expected_paths = [
      path / "1",
      path / "2",
      path / "3",
      path / "4",
      path / "5"
    ]

    self.assertTrue(all(p.exists() for p in expected_paths))
    self.assertEqual(len(input_), series)

    for ds in datasets:
      self.assertIn(ds, input_)


class HistoricInput(HistoricAbstractInput):
  required_tags = ['SOPInstanceUID']
  required_values: Dict[int|str, Any] = {
    0x0008103E : SERIES_DESCRIPTION
  }
  address = Address('localhost', 51211, "ENDPOINT")

  def thread_target(self, query_data):
    # SKIP ALL the quering, It's inside of e2e tests
    self.state = HistoricAbstractInput.HistoricInputState.FILLED

  def check_query_dataset(self, current_study: Dataset, query_dataset: Optional[Dataset] = None):
    if 'PatientID' not in current_study:
      return None

    return self.HistoricAction.FIND_QUERY, create_query_dataset(query_level=QueryLevels.PATIENT, PatientID=current_study.PatientID)


study_date = "20200101"

historic_input_dataset = Dataset()
historic_input_dataset.SOPInstanceUID = gen_uid()
historic_input_dataset.StudyDate = study_date
historic_input_dataset.PatientID = "test id"
historic_input_dataset.InstanceNumber = 1
historic_input_dataset.SeriesDescription = SERIES_DESCRIPTION

historic_input_dataset_2 = Dataset()
historic_input_dataset_2.SOPInstanceUID = gen_uid()
historic_input_dataset_2.StudyDate = study_date
historic_input_dataset_2.PatientID = "test id"
historic_input_dataset_2.InstanceNumber = 2
historic_input_dataset_2.SeriesDescription = SERIES_DESCRIPTION

historic_input_dataset_3 = Dataset()
historic_input_dataset_3.SOPInstanceUID = gen_uid()
historic_input_dataset_3.StudyDate = "20190101"
historic_input_dataset_3.PatientID = "test id"
historic_input_dataset_3.InstanceNumber = 1
historic_input_dataset_3.SeriesDescription = SERIES_DESCRIPTION

historic_input_dataset_4 = Dataset()
historic_input_dataset_4.SOPInstanceUID = gen_uid()
historic_input_dataset_4.StudyDate = "20210101"
historic_input_dataset_4.PatientID = "test id"
historic_input_dataset_4.InstanceNumber = 1
historic_input_dataset_4.SeriesDescription = SERIES_DESCRIPTION

class HistoricTestcases(DicomnodeTestCase):
  def test_historic_input_add_image(self):
    input_ = HistoricInput(config_from_raw(DicomnodeConfigRaw(AE_TITLE="BLAH BLAH")))

    self.assertEqual(0, input_.add_image(historic_input_dataset))
    self.assertEqual(0, input_.images)
    # Simulate setting from PatientNode
    input_.study_date = study_date

    if input_.thread is None:
      raise AssertionError("Thread should have been defined")

    self.assertRaises(InvalidDataset, input_.add_image, historic_input_dataset_2)
    self.assertEqual(0, input_.images)

    self.assertEqual(1, input_.add_image(historic_input_dataset_3))
    self.assertEqual(1, input_.images)

    self.assertRaises(InvalidDataset, input_.add_image, historic_input_dataset_4)
    self.assertEqual(1, input_.images)

    input_.thread.join()
    self.assertEqual(input_.state, HistoricAbstractInput.HistoricInputState.FILLED)
