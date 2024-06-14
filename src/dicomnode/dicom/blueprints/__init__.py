# Python Standard library
from io import BytesIO
from random import randint
from datetime import date, time, datetime
from typing import Any

# Third party packages
import matplotlib
matplotlib.use('agg')
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from pydicom.uid import (
  EncapsulatedPDFStorage
)

# Dicomnode packages
from dicomnode.dicom import gen_uid
from dicomnode.dicom.dicom_factory import Blueprint, CopyElement, StaticElement,\
  FunctionalElement, InstanceCopyElement, SeriesElement, InstanceEnvironment

# This trick is mostly just to prevent pollution of namespace
class _ErrorConstants():
  SAMPLES_PER_PIXEL = 3
  PHOTOMETRIC_INTERPRETATION = "RGB"
  PLANAR_CONFIGURATION = 0
  ROWS = 500
  COLUMNS = 500
  BIT_ALLOCATED = 8
  BIT_STORED = 8
  HIGH_BIT = 7
  PIXEL_REPRESENTATION = 0

ERROR_CONSTANTS = _ErrorConstants()

def generate_error_picture(_: Any):
  my_dpi = 300
  figure = Figure(figsize=(ERROR_CONSTANTS.COLUMNS / my_dpi, ERROR_CONSTANTS.ROWS / my_dpi), dpi=my_dpi)
  canvas = FigureCanvasAgg(figure)
  axis=canvas.figure.add_subplot(1,1,1)
  axis.set_axis_off()
  axis.text(50,50, "ERROR IN THE PIPELINE")
  figure.set_facecolor("red")
  canvas.draw()
  bytes_io = BytesIO()
  figure.savefig(bytes_io, format='rgba') # SO MATPLOTLIB DOESN*T SUPPORT RGB...
  bytes_io.seek(0)
  bytes_rgba = bytes_io.read()
  bytes_rgb = bytes([b for i, b in enumerate(bytes_rgba) if i % 4 != 3])
  return bytes_rgb


###### Header function ######
def _add_InstanceNumber(caller_args: InstanceEnvironment):
  # iterator is Zero indexed while, instance number is 1 indexed
  # This function assumes that the factory is aware of this
  return caller_args.instance_number

def add_UID_tag(_: Any):
  return gen_uid()

def get_today(_: Any) -> date:
  return date.today()

def get_time(_: Any) -> time:
  return datetime.now().time()

def get_now_datetime(_: Any) -> datetime:
  return datetime.now()

def get_random_number(_: Any) -> int:
  return randint(1, 2147483646)

###### Header Tag Lists ######
patient_blueprint = Blueprint([
  CopyElement(0x00100010), # PatientName
  CopyElement(0x00100020), # PatientID
  CopyElement(0x00100021, Optional=True), # Issuer of Patient ID
  CopyElement(0x00100030, Optional=True), # PatientsBirthDate
  CopyElement(0x00100040), # PatientSex
])

frame_of_reference_blueprint = Blueprint([
  CopyElement(0x00200052),
  CopyElement(0x00201040)
])

general_study_blueprint = Blueprint([
  CopyElement(0x00080020), # StudyDate
  CopyElement(0x00080030), # StudyTime
  CopyElement(0x00080050), # AccessionNumber
  CopyElement(0x00081030, Optional=True), # StudyDescription
  CopyElement(0x00200010, Optional=True), # StudyID
  CopyElement(0x0020000D), # StudyInstanceUID
])

# You might argue that you should overwrite, since this is a synthetic image
general_equipment_blueprint = Blueprint([
  CopyElement(0x00080070, Optional=True), # Manufacturer
  CopyElement(0x00080080, Optional=True), # Institution Name
  CopyElement(0x00080081, Optional=True), # Institution Address
  CopyElement(0x00081040, Optional=True), # Institution Department Name
  CopyElement(0x00081090, Optional=True), # Manufacturer's Model Name
])


general_image_blueprint = Blueprint([
  StaticElement(0x00080008, 'CS', ['DERIVED', 'PRIMARY']), # Image Type # write a test for this
  FunctionalElement(0x00200013, 'IS', _add_InstanceNumber), # InstanceNumber
])


# One might argue the optionality of these tags
image_plane_blueprint = Blueprint([
  CopyElement(0x00180050),               # Slice thickness
  InstanceCopyElement(0x00200032, 'DS'), # Image position
  CopyElement(0x00200037),               # Image Orientation
  #InstanceCopyElement(0x00201041, 'DS'), # Slice Location
  CopyElement(0x00280030),               # Pixel Spacing
])


ct_image_blueprint = Blueprint([
  CopyElement(0x00080008, Optional=True), # Image Type
  CopyElement(0x00180022, Optional=True), # Scan Options
  CopyElement(0x00180060, Optional=True), # KVP
  CopyElement(0x00180090, Optional=True), # Data Collection Diameter
  CopyElement(0x00181100, Optional=True), # Reconstruction Diameter
  CopyElement(0x00181110, Optional=True), # Distance Source to Detector
  CopyElement(0x00181111, Optional=True), # Distance Source to Patient
  CopyElement(0x00181120, Optional=True), # Gantry / Detector Tilt
  CopyElement(0x00181130, Optional=True), # Table Height
  CopyElement(0x00181140, Optional=True), # Rotation Direction
  CopyElement(0x00181150, Optional=True), # Exposure Time
  CopyElement(0x00181151, Optional=True), # X-Ray Tube Current
  CopyElement(0x00181152, Optional=True), # Exposure
  CopyElement(0x00181153, Optional=True), # Exposure in µAs
  CopyElement(0x00181160, Optional=True), # Filter Type
  CopyElement(0x00181170, Optional=True), # Generator Power
  CopyElement(0x00181190, Optional=True), # Focal Spots
  CopyElement(0x00181210, Optional=True), # Convolution Kernel
  CopyElement(0x00189305, Optional=True), # Revolution Time
])


patient_study_blueprint = Blueprint([
  CopyElement(0x00101010, Optional=True), # PatientAge
  CopyElement(0x00101020, Optional=True), # PatientSize
  CopyElement(0x00101022, Optional=True), # PatientBodyMassIndex
  CopyElement(0x00101030, Optional=True), # PatientWeight
  CopyElement(0x001021A0, Optional=True), # SmokingStatus
  CopyElement(0x001021C0, Optional=True), # PregnancyStatus
])


general_series_blueprint = Blueprint([
  CopyElement(0x00080060), # Modality
  SeriesElement(0x00080021, 'DA', get_today),         # SeriesDate
  SeriesElement(0x00080031, 'TM', get_time),          # SeriesTime
  SeriesElement(0x0020000E, 'UI', gen_uid),            # SeriesInstanceUID
  SeriesElement(0x0008103E, 'LO', lambda: "Dicomnode pipeline output"), # SeriesDescription
  SeriesElement(0x00200011, 'IS', get_random_number), # SeriesNumber
  CopyElement(0x00081070, Optional=True),              # Operators' Name
  CopyElement(0x00185100),                             # PatientPosition
])

SOP_common_blueprint: Blueprint = Blueprint([
  CopyElement(0x00080016),                                  # SOPClassUID, you might need to change this
  FunctionalElement(0x00080018, 'UI', add_UID_tag), # SOPInstanceUID
  FunctionalElement(0x00200013, 'IS', _add_InstanceNumber)  # InstanceNumber
])


default_report_blueprint = Blueprint([
  StaticElement(0x0008_0016, 'UI', EncapsulatedPDFStorage),
  CopyElement(0x00080020, Optional=True), # Study Date
  FunctionalElement(0x00080023, 'DA', get_today), #ContentDate
  FunctionalElement(0x0008002A, 'DT', get_now_datetime), # AcquisitionDateTime
  FunctionalElement(0x00080033, 'TM', get_time), #ContentTime
  CopyElement(0x00080030, Optional=True), # Study Time
  CopyElement(0x00080050), # AccessionNumber
  StaticElement(0x00080060, 'CS', 'DOC'), # Modality
  StaticElement(0x00080064, 'CS', 'WSD'), # ConversionType
  CopyElement(0x00081030), # StudyDescription
  CopyElement(0x00101010, Optional=True), #PatientAge
  CopyElement(0x00101020, Optional=True), #PatientSize
  CopyElement(0x00101030, Optional=True), #PatientWeight
  CopyElement(0x0020000D), # StudyInstanceUID
  CopyElement(0x00200010, Optional=True), # StudyID
  StaticElement(0x00200012, 'IS', 1), # InstanceNumber
  SeriesElement(0x0020000E, 'UI', add_UID_tag), # SeriesInstanceUID
  StaticElement(0x00280301, 'CS', 'NO'),  # BurnedInAnnotation
])


from .error_blueprint_english import ERROR_BLUEPRINT

__all__ = [
  'SOP_common_blueprint',
  'default_report_blueprint',
  'ERROR_BLUEPRINT'
]