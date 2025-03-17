"""This module is about manipulation of nifti objects
"""

# Python standard library
from pathlib import Path

from typing import List, Optional

# Third party packages
from pydicom import Dataset
from dicom2nifti.convert_dicom import dicom_array_to_nifti

# Dicomnode
from dicomnode.dicom import sort_datasets
from dicomnode.lib.logging import get_logger

logger = get_logger()

def convert_to_nifti(dicom_array: List[Dataset], output_file: Optional[Path], reorient: bool):
  dicom_array.sort(key=sort_datasets)

  images = dicom_array_to_nifti(dicom_array, output_file, reorient)['NII']
  return images
