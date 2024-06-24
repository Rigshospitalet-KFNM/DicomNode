"""This module is mostly for building gated pet from not gated Pets"""

# Python standard library
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

# Third party packages
from numpy import ndarray
from nibabel.loadsave import load
from nibabel.nifti1 import Nifti1Image
from pydicom import Dataset

# Dicomnode packages
from dicomnode.dicom.dicom_factory import Blueprint, DicomFactory
from dicomnode.dicom.series import DicomSeries

GATED_BLUEPRINT = Blueprint([

])

@dataclass
class BuildGatedSeriesArguments:
  images: Union[List[Path], 
                List[List[Dataset]], 
                List[Nifti1Image], 
                List[ndarray]]

def _load_nifti(nifti_image: Nifti1Image) -> ndarray:
  return nifti_image.get_fdata()

def _load_path(path) -> ndarray:
  return _load_nifti(load(path)) #type: ignore

def _load_dicom(dicoms: List[Dataset]) -> ndarray:
  if(len(dicoms) == 0):
    raise ValueError("")
  pivot = dicoms[0]
  x = pivot.Columns
  y = pivot.Rows
  z = len(dicoms)

  return ndarray((x,y,z))

def build_gated_series(
    args: BuildGatedSeriesArguments
) -> DicomSeries:
  
  images: List[ndarray] = []
  for image in args.images:
    if isinstance(image, Path):
      images.append(_load_path(image))
    if isinstance(image, Nifti1Image):
      images.append(_load_nifti(image))
    if isinstance(image, ndarray):
      images.append(image)
    if isinstance(image, List):
      images.append(_load_dicom(image))
  
  return DicomSeries([])
