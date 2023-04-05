
# Python Standard Library
import datetime
import logging
from pprint import pprint
from random import randint
import shutil
from sys import stdout
from time import sleep
from typing import Iterable, NoReturn, Optional
from unittest import skip, TestCase
import subprocess


# Third party imports
from pydicom import Dataset
from pydicom.uid import MRImageStorage, CTImageStorage
from pathlib import Path
from nibabel.nifti1 import Nifti1Image

# Dicomnode Libraries
from dicomnode.lib.dimse import send_images
from dicomnode.lib.dicom import extrapolate_image_position_patient, make_meta, gen_uid
from dicomnode.lib.exceptions import IncorrectlyConfigured
from dicomnode.lib.dicom_factory import Blueprint, FillingStrategy, patient_blueprint, frame_of_reference_blueprint, general_study_blueprint, general_equipment_blueprint,general_image_blueprint, image_plane_blueprint, patient_study_blueprint, general_series_blueprint, SOP_common_blueprint
from dicomnode.lib.numpy_factory import image_pixel_blueprint
from dicomnode.lib.nifti import NiftiGrinder, NiftiFactory
from dicomnode.server.input import AbstractInput
from dicomnode.server.nodes import AbstractPipeline
from dicomnode.server.pipeline_tree import InputContainer
from dicomnode.server.output import NoOutput, Address, FileOutput

from tests.helpers import generate_numpy_datasets, TESTING_TEMPORARY_DIRECTORY, testing_logs

class NiftyGrinderTestCase(TestCase):
  def setUp(self) -> None:
    return super().setUp()

  def tearDown(self) -> None:
    return super().tearDown()

  def test_invalid_configuration_for_grinder(self):
    self.assertRaises(IncorrectlyConfigured, NiftiGrinder, None, True)

  def test_nifti_grinder_MR_no_resampling(self):
    grinder = NiftiGrinder(None, False)

    slices = 50

    image_orientation = [1.0,0.0,0.0,0.0,1.0,0.0]

    slice_x = 2.0
    slice_y = 2.0
    slice_z = 2.0

    rows = 300
    cols = 400

    datasets = [
      ds for ds in generate_numpy_datasets(slices, Rows=rows,Cols=cols,)
    ]
    positions = extrapolate_image_position_patient(
      slice_thickness=slice_z,
      orientation=1,
      initial_position=(0.0,0.0,0.0),
      image_orientation=tuple(image_orientation),
      image_number=1,
      slices=slices
    )

    for dataset, position in zip(datasets, positions):
      dataset.ImagePositionPatient = position
      dataset.ImageOrientationPatient = image_orientation
      dataset.Modality = 'MR'
      dataset.PixelSpacing = [slice_x,slice_y]

    res = grinder(datasets)

    self.assertEqual(res.header.get_data_shape(), (cols, rows, slices)) #type: ignore

    image_data = res.get_fdata()

    self.assertTrue(image_data.flags["F_CONTIGUOUS"])
    self.assertEqual(image_data.shape, (cols, rows, slices))

  def test_nifti_grinder_MR_create_dir_and_resampling(self):
    grinder_path = Path(TESTING_TEMPORARY_DIRECTORY) / "nifti_grinder-test"
    if grinder_path.exists():
      shutil.rmtree(grinder_path) # pragma: no cover

    grinder = NiftiGrinder(grinder_path, True)

    slices = 50

    image_orientation = [1.0,0.0,0.0,0.0,1.0,0.0]

    slice_x = 2.0
    slice_y = 2.0
    slice_z = 2.0

    rows = 300
    cols = 400

    datasets = [
      ds for ds in generate_numpy_datasets(slices, Rows=rows,Cols=cols,)
    ]
    positions = extrapolate_image_position_patient(
      slice_thickness=slice_z,
      orientation=1,
      initial_position=(0.0,0.0,0.0),
      image_orientation=tuple(image_orientation),
      image_number=1,
      slices=slices
    )

    for dataset, position in zip(datasets, positions):
      dataset.ImagePositionPatient = position
      dataset.ImageOrientationPatient = image_orientation
      dataset.Modality = 'MR'
      dataset.PixelSpacing = [slice_x,slice_y]

    res = grinder(datasets)

    self.assertEqual(res.header.get_data_shape(), (cols, rows, slices)) #type: ignore

    image_data = res.get_fdata()

    # WELL WELL WELL WE HAVE SOME IDIOT ALLOCATION
    self.assertFalse(image_data.flags["F_CONTIGUOUS"])
    self.assertTrue(image_data.flags["C_CONTIGUOUS"])
    self.assertEqual(image_data.shape, (cols, rows, slices))

  def test_nifti_grinder_CT(self):
    grinder = NiftiGrinder()
    slices = 50
    image_orientation = [1.0,0.0,0.0,0.0,1.0,0.0]
    datasets = [ ds for ds in generate_numpy_datasets(slices)]
    positions = extrapolate_image_position_patient(
      slice_thickness=1,
      orientation=1,
      initial_position=(0.0,0.0,0.0),
      image_orientation=tuple(image_orientation),
      image_number=1,
      slices=slices
    )

    for dataset, position in zip(datasets, positions):
      dataset.ImagePositionPatient = position
      dataset.ImageOrientationPatient = image_orientation
      dataset.Modality = 'CT'
      dataset.PixelSpacing = [1.0,1.0]


    res = grinder(datasets)

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

  dicom_factory = NiftiFactory()
  header_blueprint = patient_blueprint \
        + frame_of_reference_blueprint \
        + general_study_blueprint \
        + general_equipment_blueprint \
        + general_image_blueprint \
        + image_plane_blueprint \
        + patient_study_blueprint \
        + general_series_blueprint \
        + SOP_common_blueprint
  filling_strategy = FillingStrategy.COPY

  # Logging
  log_level = logging.DEBUG
  log_output = None
  log_format = "%(asctime)s %(name)s %(funcName)s %(lineno)s %(levelname)s %(message)s"

  ae_title = TEST_AE_TITLE
  input = {
    INPUT_KW : NiftiInput
  }

  def process(self, input_container: InputContainer):
    nifti_object: Nifti1Image = input_container[INPUT_KW]
    data_array = nifti_object.get_fdata()
    data_array += 1

    if input_container.header is None or self.dicom_factory is None:
      raise Exception

    series = self.dicom_factory.build_from_header(
      input_container.header,
      nifti_object
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

    study_date = datetime.date(2012,8,3)
    study_time = datetime.time(11,00,00)

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

    sleep(0.25)
    node.close()




