"""This module exists to help with ensure that the dicom standard is kept.
See SOP classes in DicomForDummies.md
"""


from typing import Dict, List

from pydicom import Dataset
from pydicom.uid import (UID, CTImageStorage, EncapsulatedPDFStorage,
                         MRImageStorage, NuclearMedicineImageStorage,
                         PositronEmissionTomographyImageStorage)

CT_Image_required_tags: List[int] = [
  0x00080008, # ImageType
  0x00180060, # KVP
  0x00280100, # BitsAllocated
  0x00280101, # BitsStored,
  0x00280102, # HighBit
  0x00281052, # RescaleIntercept
  0x00281053, # RescaleType
]

Encapsulated_Document_required_tags: List[int] = [
  0x00080023, # ContentDate
  0x0008002A, # AcquisitionDateTime
  0x00080033, # ContentTime
  0x00200013, # InstanceNumber
]

Encapsulated_Document_Series_required_tags: List[int] = [
  0x00080060, # Modality
  0x0020000E, # SeriesInstanceUID
  0x00200011, # SeriesNumber
  0x00280301, # BurnedInAnnotation
  0x00420010, # DocumentTitle
  0x00420011, # EncapsulatedDocument
  0x00420012, # MIMETypeofEncapsulatedDocument
]

Frame_of_Reference_required_tags: List[int] = [
   0x00200052, # FrameOfReferenceUID
   0x00201040  # PositionReferenceIndicator
]

General_Study_required_tags: List[int] = [
  0x00080020, # StudyDate
  0x00080030, # StudyTime
  0x00080050, # AccessionNumber
  0x00080090, # ReferringPhysicianName
  0x0020000D, # StudyInstanceUID
  0x00200010, # StudyID
]

General_Series_required_tags: List[int] = [
  0x00080060, # Modality
  0x0020000E, # SeriesInstanceUID
  0x00200011, # SeriesNumber
]

General_Equipment_required_tags: List[int] = [
  0x00080070, # Manufacturer
]

General_Image_required_tags: List[int] = [
  0x00200013 # InstanceNumber
]

CR_Series: List[int] = [
  0x00180015,
  0x00185101
]

Image_Plane_required_tags: List[int] = [
  0x00180050, # Slice thickness
  0x00200032, # ImagePosition
  0x00200037,  # ImageOrientation
  0x00280030  # PixelSpacing
]

Image_Pixel_required_tags: List[int] = [
  0x00280002, # Samples per Pixel
  0x00280004, # Photometric Interpretation
  0x00280010, # Rows
  0x00280011, # Columns
  0x00280100, # BitsAllocated
  0x00280101, # BitsStored,
  0x00280102, # HighBit
  0x00280103, # PixelRepresentation
  0x7FE00010, # PixelData
]

MR_Image_required_tags: List[int] = [
  0x00080008, # ImageType
  0x00180020, # ScanningSequence
  0x00180021, # ScanningVariant
  0x00180022, # ScanOptions
  0x00180023, # MRAcquisitionType
  0x00180081, # EchoTime
  0x00280100, # BitsAllocated
  0x00280101, # BitsStored,
  0x00280102, # HighBit
]

MultiFrame_required_tags = [
  0x00280008, # NumberOfFrames
  0x00280009, # FrameIncrementPointer
]

NM_Detector_required_tags: List[int] = [
  0x00540022,  # DetectorInformationSequence
]

NM_Image_Pixel_required_tags: List[int] = [
  0x00280002, # Samples per Pixel
  0x00280004, # Photometric Interpretation
  0x00280030, # PixelSpacing
  0x00280100, # BitsAllocated
  0x00280101, # BitsStored,
  0x00280102, # HighBit
]

NM_Multi_Frame_required_tags: List[int] = [
  0x00280009, # FrameIncrementPointer
  0x00540011, #
  0x00540021,
]

NM_Image_required_tags: List[int] = [
  0x00080008 # ImageType
]

NM_Isotope_required_tags: List[int] = [
  0x00540012, # EnergyWindowInformationSequence
  0x00540016, # RadiopharmaceuticalInformationSequence
]

NM_PET_Patient_Orientation_required_tags: List[int] = [
  0x00540410, # PatientOrientationCodeSequence
  0x00540414  # PatientGantryRelationshipCodeSequence
]

RT_Image_required_tags = [
  0x00080008, # ImageType
  0x00280002, # SamplesPerPixel
  0x00280004, # PhotometricInterpretation
  0x00280100, # BitsAllocated
  0x00280101, # BitsStored,
  0x00280102, # HighBit
  0x00280103, # PixelRepresentation
  0x30020002, # RTImageLabel
  0x3002000E, # XRayImageReceptorAngle
  0x30020011, # ImagePlanePixelSpacing
  0x30020012, # RTImagePosition
  0x30020020, # RadiationMachineName
  0x30020022, # RadiationMachineSADAttribute
  0x30020026, # RTImageSID
  0x300A00B3, # PrimaryDosimeterUnit
]

RT_image = [
  0x00080008, # ImageType
  0x00080064, # ConversionType
  0x00280002, # SamplesPerPixel
  0x00280004, # PhotometricInterpretation
  0x00280100, # BitsAllocated
  0x00280101, # BitsStored,
  0x00280102, # HighBit
  0x00280103, # PixelRepresentation
  0x30020002, # RTImageLabel
  0x3002000C, # RTImagePlane
  0x3002000E, # XRayImageReceptorAngle
  0x30020011, # ImagePlanePixelSpacing
  0x30020012, # RTImagePosition,
  0x30020020, # RadiationMachineName
  0x30020022, # RadiationMachineSAD
  0x30020022, # RTImageSID
  0x300A00B3, # PrimaryDosimeterUnit
]

RT_Series_required_tags = [
  0x00080060, # Modality
  0x0020000E, # SeriesInstanceUID
  0x00200011, # SeriesNumber
]

SC_Equipment_required_tags: List[int] = [
  0x00080064 # ConversionType
]

SOP_Common_required_tags: List[int] = [
  0x00080016, # SOPInstanceUID
  0x00080018  # SOPClassUID
]

Patient_required_tags: List[int] = [
  0x00100010, # PatientName
  0x00100020, # PatientID
  0x00100030, # PatientsBirthDate
  0x00100040, # PatientSex
]

PET_Series_required_tags: List[int] = [
  0x00080021, # SeriesDate
  0x00080031, # SeriesTime
  0x00181181, # CollimatorType
  0x00280051, # CorrectedImage
  0x00540081, # NumberOfSlices
  0x00541000, # SeriesType
  0x00541001, # Units
  0x00541002, # CountsSource
  0x00541102, # DecayCorrection
]

PET_Isotope_required_tags: List[int] = [
  0x00540016, # RadiopharmaceuticalInformationSequence
]

PET_Image_required_tags: List[int] = [
  0x00080008, # ImageType
  0x00080022, # AcquisitionDate
  0x00080032, # AcquisitionTime
  0x00280002, # SamplesPerPixel
  0x00280004, # PhotometricInterpretation
  0x00280100, # BitsAllocated
  0x00280101, # BitsStored,
  0x00280102, # HighBit
  0x00281052, # RescaleIntercept
  0x00281053, # RescaleSlope
  0x00541300, # FrameReferenceTime
  0x00541330  # ImageIndex
]

CTImageStorage_required_tags: List[int] = Patient_required_tags \
  + General_Study_required_tags \
  + General_Series_required_tags \
  + Frame_of_Reference_required_tags \
  + General_Equipment_required_tags \
  + General_Image_required_tags \
  + Image_Plane_required_tags \
  + Image_Pixel_required_tags \
  + CT_Image_required_tags \
  + SOP_Common_required_tags

EncapsulatedPDFStorage_required_tags: List[int] = Patient_required_tags \
  + General_Series_required_tags \
  + Encapsulated_Document_Series_required_tags \
  + General_Equipment_required_tags \
  + SC_Equipment_required_tags \
  + Encapsulated_Document_required_tags \
  + SOP_Common_required_tags

MRImageStorage_required_tags: List[int] = Patient_required_tags \
  + General_Study_required_tags \
  + General_Series_required_tags \
  + Frame_of_Reference_required_tags \
  + General_Equipment_required_tags \
  + General_Image_required_tags \
  + Image_Plane_required_tags \
  + Image_Pixel_required_tags \
  + MR_Image_required_tags \
  + SOP_Common_required_tags

NuclearMedicineImageStorage_required_tags: List[int] = Patient_required_tags \
  + General_Study_required_tags \
  + General_Series_required_tags \
  + NM_PET_Patient_Orientation_required_tags \
  + Frame_of_Reference_required_tags \
  + General_Equipment_required_tags \
  + General_Image_required_tags \
  + Image_Pixel_required_tags \
  + NM_Image_Pixel_required_tags \
  + MultiFrame_required_tags \
  + NM_Image_required_tags \
  + NM_Isotope_required_tags \
  + NM_Detector_required_tags \
  + SOP_Common_required_tags

PositronEmissionTomographyImageStorage_required_tags: List[int] = Patient_required_tags \
  + General_Study_required_tags \
  + General_Series_required_tags \
  + PET_Series_required_tags \
  + PET_Isotope_required_tags \
  + NM_PET_Patient_Orientation_required_tags \
  + Frame_of_Reference_required_tags \
  + General_Equipment_required_tags \
  + General_Image_required_tags \
  + Image_Plane_required_tags \
  + Image_Pixel_required_tags \
  + PET_Image_required_tags \
  + SOP_Common_required_tags

DICOM_RT_Image_required_tags = Patient_required_tags \
  + General_Study_required_tags \
  + RT_Series_required_tags \
  + General_Equipment_required_tags \
  + General_Image_required_tags \
  + Image_Pixel_required_tags \
  + RT_image \
  + SOP_Common_required_tags


required_tags: Dict[UID, List[int]] = {
  CTImageStorage : CTImageStorage_required_tags,
  EncapsulatedPDFStorage : EncapsulatedPDFStorage_required_tags,
  MRImageStorage : MRImageStorage_required_tags,
  NuclearMedicineImageStorage : NuclearMedicineImageStorage_required_tags,
  PositronEmissionTomographyImageStorage : PositronEmissionTomographyImageStorage_required_tags,
}
