"""This is the test case of somebody setting up a node, and convert to nifti and
back again"""

# Python standard library
import logging
from datetime import date, time
from random import randint
from unittest.mock import patch

# Third party packages
from nibabel.nifti1 import Nifti1Image
from pydicom import Dataset
from pydicom.uid import CTImageStorage

# Dicomnode packages
from dicomnode.constants import DICOMNODE_LOGGER_NAME, DICOMNODE_PROCESS_LOGGER
from dicomnode.dicom import gen_uid, make_meta, extrapolate_image_position_patient
from dicomnode.dicom.series import NiftiSeries
from dicomnode.dicom.blueprints import add_UID_tag
from dicomnode.dicom.dicom_factory import DicomFactory, Blueprint,\
  FunctionalElement, StaticElement, SeriesElement, CopyElement
from dicomnode.dicom.dimse import Address, send_images
from dicomnode.server.processor import AbstractProcessor
from dicomnode.server.grinders import NiftiGrinder
from dicomnode.server.input import AbstractInput
from dicomnode.server.nodes import AbstractPipeline
from dicomnode.server.output import FileOutput, NoOutput, PipelineOutput, DicomOutput
from dicomnode.server.input_container import InputContainer

# Testing packages
from tests.helpers import TESTING_TEMPORARY_DIRECTORY, generate_numpy_datasets,\
  process_thread_check_leak, clear_logger
from tests.helpers.dicomnode_test_case import DicomnodeTestCase
from tests.helpers.storage_endpoint import TestStorageEndpoint, ENDPOINT_AE_TITLE, ENDPOINT_PORT

blueprint = Blueprint([
  StaticElement(0x0008_0016, 'UI', CTImageStorage),
  FunctionalElement(0x0008_0018,'UI', add_UID_tag),
  CopyElement(0x0010_0010),
  CopyElement(0x0010_0020),
  SeriesElement(0x0020_000D,'UI', add_UID_tag),
  SeriesElement(0x0020_000E,'UI', add_UID_tag),
])

##### Constants #####
TEST_AE_TITLE = "NIFTYAE"
SENDER_AE = "SENDERAE"
INPUT_KW = "input"

class NiftiProcessor(AbstractProcessor):
  def process(self, input_container: InputContainer) -> PipelineOutput:
    nifti_object: Nifti1Image = input_container[INPUT_KW]
    series = NiftiSeries(nifti_object)
    data_array = series.image.raw
    data_array += 1

    dicom_factory = DicomFactory()

    series = dicom_factory.build_series(
      data_array,
      blueprint,
      input_container.datasets[INPUT_KW],
    )

    self.logger.debug("Build the thing")

    return DicomOutput([(Address("127.0.0.1", ENDPOINT_PORT, ENDPOINT_AE_TITLE), series)], TEST_AE_TITLE)

class NiftiInput(AbstractInput):
    required_tags = [
      0x0008_0016, # SOPClassUID
      0x0008_0018, # SOPInstanceUID
      0x0020_0013, # InstanceNumber
      0x0008_0060, # Modality
      0x0018_0050, # SliceThickness
      0x0020_0032, # ImagePosition
      0x0020_0037, # ImageOrientationPatient
      0x0028_0010, # Rows,
      0x0028_0011, # Columns,
      0x0028_0030, # PixelSpacing,
      0x7FE0_0010, # PixelData
    ]
    def validate(self) -> bool:
      return True

    image_grinder = NiftiGrinder()


class NiftiNode(AbstractPipeline):
  # Directories
  # Logging
  log_level = logging.DEBUG
  log_output = 'nifti_end2end'
  log_format = "%(asctime)s %(name)s %(funcName)s %(lineno)s %(levelname)s %(message)s"

  ae_title = TEST_AE_TITLE
  input = {
    INPUT_KW : NiftiInput
  }

  Processor = NiftiProcessor

class End2EndNiftiTestCase(DicomnodeTestCase):
  def tearDown(self) -> None:
    clear_logger(DICOMNODE_LOGGER_NAME)
    clear_logger(DICOMNODE_PROCESS_LOGGER)
    return super().tearDown()

  #@skip("This caused a cascade of bugs")
  @process_thread_check_leak
  def test_end2end_nifti_plus(self):
    endpoint = TestStorageEndpoint()
    endpoint.open()

    node = NiftiNode()
    port = randint(1025, ENDPOINT_PORT - 1)
    node.port = port

    with patch('dicomnode.lib.logging.set_logger'):
      with self.assertLogs(DICOMNODE_LOGGER_NAME):
        node.open(blocking=False)

        address = Address('localhost', port, TEST_AE_TITLE)
        slices = 10
        t_image_orientation = (1.0,0.0,0.0,0.0,1.0,0.0)
        image_orientation = [1.0,0.0,0.0,0.0,1.0,0.0]

        slice_x = 2.0
        slice_y = 2.0
        slice_z = 2.0

        rows = 30
        cols = 40

        PatientID = "FooBar"
        datasets = [ ds for ds in generate_numpy_datasets(
          slices,
          Rows=rows,
          Cols=cols,
          PatientID=PatientID
        )]

        positions = extrapolate_image_position_patient(
          slice_thickness=slice_z,
          orientation=1,
          initial_position=(0.0,0.0,0.0),
          image_orientation=t_image_orientation,
          image_number=1,
          slices=slices
        )

        frame_of_reference_uid = gen_uid()

        study_date = date(2012,8,3)
        study_time = time(11,00,00)

        for i, (dataset, position) in enumerate(zip(datasets, positions)):
          dataset.InstanceNumber = i + 1
          dataset.PatientName = "Test^Person^1"
          dataset.PatientSex = "M"
          dataset.AccessionNumber = "AccessionNumber"
          dataset.StudyDate = study_date
          dataset.StudyTime = study_time
          dataset.SliceThickness=slice_z
          dataset.FrameOfReferenceUID = frame_of_reference_uid
          dataset.PositionReferenceIndicator = None
          dataset.ImagePositionPatient = position
          dataset.ImageOrientationPatient = image_orientation
          dataset.PatientPosition = "FFS"
          dataset.Modality = "CT"
          dataset.PixelSpacing = [slice_x,slice_y]
          dataset.SOPClassUID = CTImageStorage
          make_meta(dataset)

        send_images(TEST_AE_TITLE, address, datasets)

        node.close()
        endpoint.close()
