"""
This file contains a blueprint instance which is used for reporting unhandled
errors in english.
"""

# Python standard library

# Third party packages
from pydicom.uid import SecondaryCaptureImageStorage

# Dicomnode packages
from dicomnode.dicom.blueprints import get_today, get_time, add_UID_tag,\
  get_random_number, ERROR_CONSTANTS, generate_error_picture
from dicomnode.dicom.dicom_factory import Blueprint, FunctionalElement,\
  CopyElement, StaticElement, SeriesElement

ERROR_BLUEPRINT = Blueprint([
  # Patient
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
  StaticElement(0x0008_103E, 'LO', 'ERROR IN PIPELINE PROCESSING!'), # Series Description
  SeriesElement(0x0020_000E, 'UI', add_UID_tag), # Series UID
  SeriesElement(0x0020_0011, 'IS', get_random_number), # Series Number
  # SC Equipment
  StaticElement(0x0008_0064, 'CS', 'SYN'), # Conversion Type
  # General Image
  StaticElement(0x0008_0008, 'CS', ['ORIGINAL', 'SECONDARY']), # Image type
  StaticElement(0x0020_0013, 'IS', 1), # Instance Number
  # Image Pixel
  StaticElement(0x0028_0002, 'US', ERROR_CONSTANTS.SAMPLES_PER_PIXEL), # SamplesPerPixel
  StaticElement(0x0028_0004, 'CS', ERROR_CONSTANTS.PHOTOMETRIC_INTERPRETATION), # Photometric Interpretation
  StaticElement(0x0028_0006, 'US', ERROR_CONSTANTS.PLANAR_CONFIGURATION), # Planar Configuration
  StaticElement(0x0028_0010, 'US', ERROR_CONSTANTS.ROWS), # Rows
  StaticElement(0x0028_0011, 'US', ERROR_CONSTANTS.COLUMNS), # Columns
  StaticElement(0x0028_0100, 'US', ERROR_CONSTANTS.BIT_ALLOCATED), # Bits Allocated
  StaticElement(0x0028_0101, 'US', ERROR_CONSTANTS.BIT_STORED), # Bits Allocated
  StaticElement(0x0028_0102, 'US', ERROR_CONSTANTS.HIGH_BIT), # High Bit
  StaticElement(0x0028_0103, 'US', ERROR_CONSTANTS.PIXEL_REPRESENTATION), # Pixel Representation
  FunctionalElement(0x7FE0_0010, 'OB', generate_error_picture),
  #SOP Common
  StaticElement(0x0008_0016, 'UI', SecondaryCaptureImageStorage),
  FunctionalElement(0x0008_0018, 'UI', add_UID_tag),
])
"""Blueprint for unhandled exceptions, produces a Secondary Capture Image from
a dicom series"""
