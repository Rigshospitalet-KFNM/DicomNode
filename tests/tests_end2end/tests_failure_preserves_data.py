# Python Standard Library
import os
from pathlib import Path
from random import randint
from shutil import rmtree as rmdir
from unittest import skip

# Third party
from pydicom import Dataset
from pydicom.uid import SecondaryCaptureImageStorage

# Dicomnode Modules
from dicomnode.constants import DICOMNODE_LOGGER_NAME
from dicomnode.config import DicomnodeConfig
from dicomnode.dicom import gen_uid, make_meta
from dicomnode.dicom.dimse import Address, send_image
from dicomnode.lib.logging import LoggerConfig
from dicomnode.server.nodes import AbstractPipeline
from dicomnode.server.input import AbstractInput
from dicomnode.server.input_container import InputContainer
from dicomnode.server.output import PipelineOutput
from dicomnode.server.processor import AbstractProcessor
from dicomnode.server.grinders import DicomTreeGrinder

# Test Modules
from tests.helpers.dicomnode_test_case import DicomnodeTestCase

class HungryInput(AbstractInput):
  def _get_storage_type(self, config: DicomnodeConfig):
    storage_type = super()._get_storage_type(config)

    self.logger.info(f"Input have: {storage_type.__name__} storage type")

    return storage_type

  image_grinder = DicomTreeGrinder()

  def validate(self) -> bool:
    return self.images > 0

class KABOOMProcessor(AbstractProcessor):
  def process(self, input_container: InputContainer) -> PipelineOutput:
    raise Exception("KABOOOOOOOM")

class E2EFailurePreserving(DicomnodeTestCase):
  def tearDown(self) -> None:
    test_path = Path(os.getcwd()) / self._testMethodName
    if test_path.exists():
      rmdir(test_path)


  def test_e2e_preserve_datasets_if_processor_dies_with_processing_dir(self):
    """This tests check that if you have a pipeline that fails in the pipeline"""
    data_archive = Path(os.getcwd()) / self._testMethodName / "data_archive"
    processing = Path(os.getcwd()) / self._testMethodName / "processing"


    data_archive.mkdir(parents=True)

    class Pipeline(AbstractPipeline):
      ae_title = "AE_TITLE"

      log_output = None

      port = 0
      input = {
        "INPUT" : HungryInput
      }

      Processor = KABOOMProcessor

      data_directory = data_archive
      processing_directory = processing

    dataset = Dataset()
    dataset.SOPInstanceUID = gen_uid()
    dataset.StudyInstanceUID = gen_uid()
    dataset.SeriesInstanceUID = gen_uid()
    dataset.SOPClassUID = SecondaryCaptureImageStorage
    dataset.PatientID = "TEST"
    make_meta(dataset)

    p = Pipeline()
    with self.assertLogs(DICOMNODE_LOGGER_NAME):

      p.open(False)

      local_address = Address('localhost', p.get_server_port(), p.ae_title)
      send_image("TESTER", local_address, dataset)

      p.close()

    files = sum([1 for _ in data_archive.glob('*')])
    self.assertEqual(files, 1)

  def test_e2e_preserve_datasets_if_processor_dies_without_processing_dir(self):
    """This tests check that if you have a pipeline that fails in the pipeline"""
    data_archive = Path(os.getcwd()) / self._testMethodName / "data_archive"
    data_archive.mkdir(parents=True)

    class Pipeline(AbstractPipeline):
      ae_title = "AE_TITLE"
      log_output = None

      port = 0
      input = {
        "INPUT" : HungryInput
      }

      Processor = KABOOMProcessor

      data_directory = data_archive

    dataset = Dataset()
    dataset.SOPInstanceUID = gen_uid()
    dataset.StudyInstanceUID = gen_uid()
    dataset.SeriesInstanceUID = gen_uid()
    dataset.SOPClassUID = SecondaryCaptureImageStorage
    dataset.PatientID = "TEST"
    make_meta(dataset)

    p = Pipeline()
    with self.assertLogs(DICOMNODE_LOGGER_NAME):

      p.open(False)
      local_address = Address('localhost', p.get_server_port(), p.ae_title)


      send_image("TESTER", local_address, dataset)

      p.close()

    files = sum([1 for _ in data_archive.glob('*')])
    self.assertEqual(files, 1)