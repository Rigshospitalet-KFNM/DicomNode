"""This module is mostly for building gated pet from not gated Pets"""

# Python standard library
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

# Third party packages
from numpy import concatenate, ndarray
from nibabel.loadsave import load
from nibabel.nifti1 import Nifti1Image
from pydicom import Dataset

# Dicomnode packages
from dicomnode.dicom.dicom_factory import Blueprint, DicomFactory
from dicomnode.dicom.series import DicomSeries
from dicomnode.math.affine import AffineMatrix
from dicomnode.math.image import build_image_from_datasets

GATED_BLUEPRINT = Blueprint([

])

@dataclass
class BuildGatedSeriesArguments:
  images: Union[List[Path], 
                List[List[Dataset]], 
                List[Nifti1Image], 
                List[ndarray]]

def _load_affine_nifti(nifti_image: Nifti1Image):
  return AffineMatrix.from_nifti(nifti_image)

def _load_affine_path(path: Path):
  return _load_affine_nifti(load(path))

def _load_nifti(nifti_image: Nifti1Image) -> ndarray:
  return nifti_image.get_fdata()

def _load_path(path: Path) -> ndarray:
  return _load_nifti(load(path)) #type: ignore

def _load_dicom(dicoms: List[Dataset]) -> ndarray:
  return build_image_from_datasets(dicoms)

def build_gated_series(
    args: BuildGatedSeriesArguments
) -> DicomSeries:
  
  images: List[ndarray] = []
  pivot = args.images[0]

  if isinstance(pivot, Path):
    affine = _load_affine_path(pivot)
  elif isinstance(pivot, Nifti1Image):
    affine = _load_nifti(pivot)
  elif isinstance(pivot, List):
    pass

  for image in args.images:
    if isinstance(image, Path):
      images.append(_load_path(image))
    elif isinstance(image, Nifti1Image):
      images.append(_load_nifti(image))
    elif isinstance(image, ndarray):
      images.append(image)
    elif isinstance(image, List):
      images.append(_load_dicom(image))
  
  image_space = concatenate(images)

  return DicomSeries([])
