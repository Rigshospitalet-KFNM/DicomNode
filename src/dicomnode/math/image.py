"""This module concerns itself with an 'image' which for this module is the raw
data"""

# Python standard library
from typing import Any, List, Literal, Tuple, TypeAlias, Union

# Third party Packages
from numpy import append, array, ceil, empty, float32, float64, floor, identity, int32, ndarray, zeros_like
from numpy.linalg import inv
from pydicom import Dataset

# Dicomnode packages
from dicomnode.constants import UNSIGNED_ARRAY_ENCODING, SIGNED_ARRAY_ENCODING
from dicomnode.lib.exceptions import InvalidDataset
from dicomnode.math.affine import Space

numpy_image: TypeAlias = ndarray[Tuple[int,int,int], Any]
raw_image_frames: TypeAlias = ndarray[Tuple[int,int,int,int], Any]

def fit_image_into_unsigned_bit_range(image: ndarray,
                                      bits_stored = 16,
                                      bits_allocated = 16,
                                     ) -> Tuple[numpy_image, float, float]:
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

def build_image_from_datasets(datasets: List[Dataset]) -> numpy_image:
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

    image_array: numpy_image = empty((z_dim, y_dim, x_dim), dtype=dataType)

    for i, dataset in enumerate(datasets):
      image = dataset.pixel_array
      if rescale:
        image = image.astype(dataType) * dataset.RescaleSlope\
              + dataset.RescaleIntercept
      image_array[i,:,:] = image

    return image_array

class Image:
  def __init__(self,
               image_data: numpy_image,
               affine: Space,
               minimum_value=0) -> None:
    self.raw = image_data
    self.affine = affine
    self.minimum_value = minimum_value


  @classmethod
  def from_datasets(cls, datasets: List[Dataset]):
    image_data = build_image_from_datasets(datasets)
    affine = Space.from_datasets(datasets)

    return cls(image_data, affine)

  @staticmethod
  def _map_args_coordinates(*args) -> ndarray[Tuple[Literal[4]], Any]:
        # Args manipulation
    input_arg = args[0]
    if len(input_arg) == 3:
      i,j,k = input_arg
      coordinates = array([i,j,k])
    elif len(args) == 1:
      if isinstance(input_arg, Tuple) or isinstance(input_arg, List):
        i,j,k = tuple(input_arg[0])
        coordinates = array([i,j,k])
      else:
        coordinates = args[0]
    else:
      error_message = f"Invalid number of args. Must be 1 or 3, but got {len(args)}"
      raise TypeError(error_message)

    if not isinstance(coordinates, ndarray): #pragma: no cover
      raise TypeError("Could not converts arguments to a Numpy.ndarray")

    if coordinates.shape != (3,): #pragma: no cover
      raise ValueError("Input vector is not a coordinate (x,y,z)")

    return coordinates

  @staticmethod
  def _map_args_point(*args) -> ndarray[Tuple[Literal[3]], Any]:
        # Args manipulation
    if len(args) == 3:
      i,j,k = args
      coordinates = array([i,j,k])
    elif len(args) == 1:
      if isinstance(args[0], Tuple) or isinstance(args[0], List):
        i,j,k = args[0]
        coordinates = array([i,j,k])
      else:
        coordinates = args[0]
    else:
      error_message = f"Invalid number of args. Must be 1 or 3, but got {len(args)}"
      raise TypeError(error_message)

    if not isinstance(coordinates, ndarray):
      raise TypeError("Could not converts arguments to a Numpy.ndarray")

    return coordinates

  def _get_value_box_around_index(self, pseudo_index):
    x,y,z = pseudo_index

    fx, fy, fz = floor(pseudo_index).astype(int32)
    cx, cy, cz = ceil(pseudo_index).astype(int32)

    dis_fx = (fx - x) ** 2
    dis_fy = (fy - y) ** 2
    dis_fz = (fz - z) ** 2

    dis_cx = (cx - x) ** 2
    dis_cy = (cy - y) ** 2
    dis_cz = (cz - z) ** 2

    return [
      (dis_fx + dis_fy + dis_fz, self.value_at_index(fx,fy,fz)),
      (dis_cx + dis_fy + dis_fz, self.value_at_index(cx,fy,fz)),
      (dis_fx + dis_cy + dis_fz, self.value_at_index(fx,cy,fz)),
      (dis_cx + dis_cy + dis_fz, self.value_at_index(cx,cy,fz)),
      (dis_fx + dis_fy + dis_cz, self.value_at_index(fx,fy,cz)),
      (dis_cx + dis_fy + dis_cz, self.value_at_index(cx,fy,cz)),
      (dis_fx + dis_cy + dis_cz, self.value_at_index(fx,cy,cz)),
      (dis_cx + dis_cy + dis_cz, self.value_at_index(cx,cy,cz)),
    ]

  def _pseudo_index_at_point(self, *args):
    """_summary_

    Returns:
        _type_: _description_
    """
    point = self._map_args_point(args)
    return self.affine.inverted_raw @ point

  def value_at_index(self, *args):
    x, y, z = self._map_args_point(args)
    columns, rows, slices = self.raw.shape

    if not (0 < x < columns):
      return self.minimum_value

    if not (0 < y < rows):
      return self.minimum_value

    if not (0 < z < slices):
      return self.minimum_value

    return self.raw[x,y,z]

  def value_at_point_nn(self, *args):
    minimum_distance = 1000
    return_value = self.minimum_value
    pseudo_index = self._pseudo_index_at_point(args)

    for (distance, value) in self._get_value_box_around_index(pseudo_index):
      if distance < minimum_distance:
        minimum_distance = distance
        return_value = value

    return return_value

  def center_index(self):
    columns, rows, slices = self.raw.shape

    return (
      (columns - 1) // 2,
      (rows - 1) // 2,
      (slices - 1) // 2,
    )

class FramedImage():
  def __init__(self, frames: raw_image_frames) -> None:
    self.raw = frames