"""This is the test case of somebody setting up a node, and convert to nifti and
back again"""

# Python standard library
import logging
from datetime import date, time
from pathlib import Path
from random import randint
import shutil
from typing import Iterable, NoReturn, Optional
from time import sleep
from unittest import TestCase

# Third party packages
from nibabel.nifti1 import Nifti1Image
from pydicom import Dataset
from pydicom.uid import CTImageStorage

# Dicomnode packages
from dicomnode.constants import DICOMNODE_LOGGER_NAME
from dicomnode.dicom import gen_uid, make_meta, extrapolate_image_position_patient
from dicomnode.dicom.dicom_factory import DicomFactory, Blueprint
from dicomnode.dicom.dimse import Address, send_images
from dicomnode.server.grinders import NiftiGrinder
from dicomnode.server.input import AbstractInput
from dicomnode.server.nodes import AbstractPipeline
from dicomnode.server.output import FileOutput
from dicomnode.server.pipeline_tree import InputContainer

# Testing packages
from tests.helpers import TESTING_TEMPORARY_DIRECTORY, generate_numpy_datasets


blueprint = Blueprint([])

##### Constants #####
TEST_AE_TITLE = "NIFTYAE"
SENDER_AE = "SENDERAE"
INPUT_KW = "input"

class NiftiInput(AbstractInput):
    required_tags = [
      0x00200013, # InstanceNumber
      0x00080060, # Modality
      0x00180050, # SliceThickness
      0x00200032, # ImagePosition
      0x00200037, # ImageOrientationPatient
      0x00280010, # Rows,
      0x00280011, # Columns,
      0x00280030, # PixelSpacing,
      0x7FE00010, # PixelData
    ]
    def validate(self) -> bool:
      return True

    image_grinder = NiftiGrinder(Path(TESTING_TEMPORARY_DIRECTORY) / "nifti_test_grinder_node", reorient_nifti=True)


def save_instance_number(path: Path, datasets: Iterable[Dataset]):
  pass

class NiftiNode(AbstractPipeline):
  # Directories
  output_dir = Path(f"{TESTING_TEMPORARY_DIRECTORY}/output")
  processing_directory = Path(f"{TESTING_TEMPORARY_DIRECTORY}/nifty_working_dir")

  # Logging
  log_level = logging.DEBUG
  log_output = None
  log_format = "%(asctime)s %(name)s %(funcName)s %(lineno)s %(levelname)s %(message)s"

  ae_title = TEST_AE_TITLE
  input = {
    INPUT_KW : NiftiInput
  }

  def process(self, input_data: InputContainer):
    nifti_object: Nifti1Image = input_data[INPUT_KW]
    data_array = nifti_object.get_fdata()
    data_array += 1

    dicom_factory = DicomFactory()

    series = dicom_factory.build_series(
      data_array,
      blueprint,
      input_data.datasets[INPUT_KW],
    )

    return FileOutput(
      [(self.output_dir, series)]
    )

  def open(self, blocking=True) -> Optional[NoReturn]:
    if not self.output_dir.exists():
      self.output_dir.mkdir(exist_ok=True)
    return super().open(blocking)

  def close(self) -> None:
    try:
      shutil.rmtree(self.output_dir)
    except Exception as E:
      self.logger.error(E)
    return super().close()


class End2EndNiftiTestCase(TestCase):
  #@skip("This caused a cascade of bugs")
  def test_end_to_end_plus(self):
    node = NiftiNode()
    port = randint(1025, 65535)
    node.port = port
    node.open(blocking=False)

    address = Address('localhost', port, TEST_AE_TITLE)
    slices = 50

    image_orientation = [1.0,0.0,0.0,0.0,1.0,0.0]

    slice_x = 2.0
    slice_y = 2.0
    slice_z = 2.0

    rows = 300
    cols = 400

    PatientID = "FooBar"
    datasets = [ ds for ds in generate_numpy_datasets(slices,
                                                      Rows=rows,
                                                      Cols=cols,
                                                      PatientID=PatientID
                                                      )]
    positions = extrapolate_image_position_patient(
      slice_thickness=slice_z,
      orientation=1,
      initial_position=(0.0,0.0,0.0),
      image_orientation=tuple(image_orientation),
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

    with self.assertLogs(DICOMNODE_LOGGER_NAME) as captured_logs:
      send_images(TEST_AE_TITLE, address, datasets)
      node.close()

    print(captured_logs)
