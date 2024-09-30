""""""

# Python standard Library

# Third party packages
from pydicom.uid import SecondaryCaptureImageStorage

from PIL.Image import Image

# Dicomnode package
from dicomnode.dicom.blueprints import Blueprint, CopyElement, StaticElement,\
  get_time, get_today, SeriesElement, FunctionalElement, add_UID_tag,\
  get_random_number
from dicomnode.dicom.dicom_factory import InstanceEnvironment


class SECONDARY_IMAGE_REPORT_CONSTANTS():
  SAMPLES_PER_PIXEL = 3
  PHOTOMETRIC_INTERPRETATION = "RGB"
  PLANAR_CONFIGURATION = 0
  BIT_ALLOCATED = 8
  BIT_STORED = 8
  HIGH_BIT = 7
  PIXEL_REPRESENTATION = 0

def generate_report_image(env: InstanceEnvironment):
  report_page_image: Image = env.kwargs['__dicom_factory_image']

  return report_page_image.tobytes('rgb')

def get_image_rows(env: InstanceEnvironment):
  report_page_image: Image = env.kwargs['__dicom_factory_image']

  return report_page_image.width

def get_image_cols(env: InstanceEnvironment):
  report_page_image: Image = env.kwargs['__dicom_factory_image']

  return report_page_image.height

def get_instance_number(env: InstanceEnvironment):
  return env.kwargs['__dicom_factory_page_number']

SECONDARY_IMAGE_REPORT_BLUEPRINT = Blueprint([
  CopyElement(0x0010_0010), # Patient Name
  CopyElement(0x0010_0020), # Patient ID
  CopyElement(0x0010_0030, Optional=True), # Patient Birth date
  CopyElement(0x0010_0032, Optional=True), # Patient Birth time
  CopyElement(0x0010_0040, Optional=True), # Patient Sex
  # General Study
  CopyElement(0x0008_0020, Optional=True), # Study Date
  CopyElement(0x0008_0030, Optional=True), # Study time
  CopyElement(0x0008_0050), # Accession Number
  CopyElement(0x0008_1030, Optional=True), # Study Description
  CopyElement(0x0020_000D), # Study UID
  CopyElement(0x0020_0010, Optional=True), # Study ID
  # Patient Study
  CopyElement(0x0010_1010, Optional=True), # Patient Size
  CopyElement(0x0010_1030, Optional=True), # Patient Weight
  # General Series
  SeriesElement(0x0008_0021, 'DA', get_today), # Series Date
  SeriesElement(0x0008_0031, 'TM', get_time), # Series Time
  StaticElement(0x0008_0060, 'CS', 'OT'), # Modality
  StaticElement(0x0008_103E, 'LO', 'Study report'), # Series Description
  SeriesElement(0x0020_000E, 'UI', add_UID_tag), # Series UID
  SeriesElement(0x0020_0011, 'IS', get_random_number), # Series Number
  # SC Equipment
  StaticElement(0x0008_0064, 'CS', 'SYN'), # Conversion Type
  # General Image
  StaticElement(0x0008_0008, 'CS', ['ORIGINAL', 'SECONDARY']), # Image type
  FunctionalElement(0x0020_0013, 'IS', get_instance_number), # Instance Number
  # Image Pixel
  StaticElement(0x0028_0002, 'US', SECONDARY_IMAGE_REPORT_CONSTANTS.SAMPLES_PER_PIXEL), # SamplesPerPixel
  StaticElement(0x0028_0004, 'CS', SECONDARY_IMAGE_REPORT_CONSTANTS.PHOTOMETRIC_INTERPRETATION), # Photometric Interpretation
  StaticElement(0x0028_0006, 'US', SECONDARY_IMAGE_REPORT_CONSTANTS.PLANAR_CONFIGURATION), # Planar Configuration
  FunctionalElement(0x0028_0010, 'US', get_image_rows), # Rows
  FunctionalElement(0x0028_0011, 'US', get_image_cols), # Columns
  StaticElement(0x0028_0100, 'US', SECONDARY_IMAGE_REPORT_CONSTANTS.BIT_ALLOCATED), # Bits Allocated
  StaticElement(0x0028_0101, 'US', SECONDARY_IMAGE_REPORT_CONSTANTS.BIT_STORED), # Bits Allocated
  StaticElement(0x0028_0102, 'US', SECONDARY_IMAGE_REPORT_CONSTANTS.HIGH_BIT), # High Bit
  StaticElement(0x0028_0103, 'US', SECONDARY_IMAGE_REPORT_CONSTANTS.PIXEL_REPRESENTATION), # Pixel Representation
  FunctionalElement(0x7FE0_0010, 'OB', generate_report_image),
  #SOP Common
  StaticElement(0x0008_0016, 'UI', SecondaryCaptureImageStorage),
  FunctionalElement(0x0008_0018, 'UI', add_UID_tag),
])