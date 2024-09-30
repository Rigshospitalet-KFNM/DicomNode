"""This module handles rt struct.

It mostly relies on rt_utils, however that library have a bug, where if you have
a single point contour, open cv throws an unhandled exception,

"""
# Python standard library
from typing import List

# Third party packages
import numpy
from pydicom import Dataset, Sequence
from rt_utils import RTStruct
from rt_utils.image_helper import create_empty_series_mask,\
  get_patient_to_pixel_transformation_matrix,\
  get_slice_contour_data, get_slice_mask_from_slice_contour_data
from rt_utils.ds_helper import get_contour_sequence_by_roi_number

# Dicomnode packages
from dicomnode.lib.exceptions import InvalidDataset


def get_contour_sequence_by_name(RT_dataset: Dataset, name: str) -> Sequence:
  if 'StructureSetROISequence' not in RT_dataset:
    raise InvalidDataset("RT data doesn't contain any contour data (StructureSetROISequence)")

  for roi in RT_dataset.StructureSetROISequence:
    if roi.ROIName == name:
      return get_contour_sequence_by_roi_number(RT_dataset, roi.ROINumber)

  raise InvalidDataset(f"RT struct doesn't contain any contour named {name}")

def get_mask_ds(series: List[Dataset], RT_dataset: Dataset, name: str) -> numpy.ndarray:
  contour_sequence = get_contour_sequence_by_name(RT_dataset, name)
  mask = create_empty_series_mask(series)
  transformation_matrix = get_patient_to_pixel_transformation_matrix(series)

  for i, series_slice in enumerate(series):
    contour_slices = [
      slice_contour_data
        for slice_contour_data in get_slice_contour_data(series_slice, contour_sequence)
          if len(slice_contour_data) > 3 # This is the line that prevent the error from happening
    ]
    if len(contour_slices):
      mask[:,:,i] = get_slice_mask_from_slice_contour_data(
        series_slice, contour_slices, transformation_matrix
      )

  return mask

def get_mask(rt_struct: RTStruct, name: str) -> numpy.ndarray:
  return get_mask_ds(rt_struct.series_data, rt_struct.ds, name)
