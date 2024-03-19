from dicomnode.dicom import gen_uid
from dicomnode.dicom.dicom_factory import Blueprint, CopyElement, StaticElement,\
  FunctionalElement, InstanceCopyElement, SeriesElement, InstanceEnvironment
from random import randint
from datetime import date, time, datetime

###### Private functions

###### Header function ######

def _add_InstanceNumber(caller_args: InstanceEnvironment):
  # iterator is Zero indexed while, instance number is 1 indexed
  # This function assumes that the factory is aware of this
  return caller_args.instance_number

def _add_UID_tag(_: InstanceEnvironment):
  return gen_uid()

def _get_today(_: InstanceEnvironment) -> date:
  return date.today()

def _get_time(_: InstanceEnvironment) -> time:
  return datetime.now().time()

def _get_now_datetime(_: InstanceEnvironment) -> datetime:
  return datetime.now()

def _get_random_number(_:InstanceEnvironment) -> int:
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
  SeriesElement(0x00080021, 'DA', _get_today),         # SeriesDate
  SeriesElement(0x00080031, 'TM', _get_time),          # SeriesTime
  SeriesElement(0x0020000E, 'UI', gen_uid),            # SeriesInstanceUID
  SeriesElement(0x0008103E, 'LO', lambda: "Dicomnode pipeline output"), # SeriesDescription
  SeriesElement(0x00200011, 'IS', _get_random_number), # SeriesNumber
  CopyElement(0x00081070, Optional=True),              # Operators' Name
  CopyElement(0x00185100),                             # PatientPosition
])

SOP_common_blueprint: Blueprint = Blueprint([
  CopyElement(0x00080016),                                  # SOPClassUID, you might need to change this
  FunctionalElement(0x00080018, 'UI', _add_UID_tag), # SOPInstanceUID
  FunctionalElement(0x00200013, 'IS', _add_InstanceNumber)  # InstanceNumber
])


default_report_blueprint = Blueprint([
  CopyElement(0x00080020, Optional=True), # Study Date
  FunctionalElement(0x00080023, 'DA', _get_today), #ContentDate
  FunctionalElement(0x0008002A, 'DT', _get_now_datetime), # AcquisitionDateTime
  FunctionalElement(0x00080033, 'TM', _get_time), #ContentTime
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
  SeriesElement(0x0020000E, 'UI', _add_UID_tag), # SeriesInstanceUID
  StaticElement(0x00280301, 'CS', 'NO'),  # BurnedInAnnotation
])
