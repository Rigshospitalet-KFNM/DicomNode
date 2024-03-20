"""This module concerns itself with an 'image' which for this module is the raw
data"""

# Python standard library
from typing import Any, List, Tuple, TypeAlias

# Third party Packages
from numpy import empty, float32, float64, ndarray, zeros_like
from pydicom import Dataset

# Dicomnode packages
from dicomnode.constants import UNSIGNED_ARRAY_ENCODING, SIGNED_ARRAY_ENCODING
from dicomnode.lib.exceptions import InvalidDataset

image_type: TypeAlias = ndarray[Tuple[int,int,int], Any]

def fit_image_into_unsigned_bit_range(image: image_type,
                                      bits_stored = 16,
                                      bits_allocated = 16,
                                     ) -> Tuple[image_type, float, float]:
    target_datatype = UNSIGNED_ARRAY_ENCODING.get(bits_allocated, None)
    min_val = image.min()
    max_val = image.max()

    if max_val == min_val:
      return zeros_like(image), 1.0, min_val

    image_max_value = ((1 << bits_stored) - 1)

    slope = (max_val - min_val) / image_max_value
    intercept = min_val

    new_image = ((image - intercept) / slope).astype(target_datatype)

    return new_image, slope, intercept

def build_image_from_datasets(datasets: List[Dataset]) -> image_type:
    pivot = datasets[0]
    x_dim = pivot.Columns
    y_dim = pivot.Rows
    z_dim = len(datasets)
    # tags are RescaleIntercept, RescaleSlope
    rescale = (0x00281052 in pivot and 0x00281053 in pivot) 

    if 0x7FE00008 in pivot:
      dataType = float32
    elif 0x7FE00009 in pivot:
      dataType = float64
    elif rescale:
      dataType = float64
    elif pivot.PixelRepresentation == 0:
      dataType = UNSIGNED_ARRAY_ENCODING.get(pivot.BitsAllocated, None)
    else:
      dataType = SIGNED_ARRAY_ENCODING.get(pivot.BitsAllocated, None)

    if dataType is None:
      raise InvalidDataset

    image_array: image_type = empty((z_dim, y_dim, x_dim), dtype=dataType)

    for i, dataset in enumerate(datasets):
      image = dataset.pixel_array
      if rescale:
        image = image.astype(float64) * dataset.RescaleSlope\
              + dataset.RescaleIntercept
      image_array[i,:,:] = image

    return image_array
