"""This module concerns itself with an 'image' which for this module is the raw
data"""

# Python standard library
from typing import Any, List, Literal, Tuple, TypeAlias, Union

# Third party Packages
import numpy
from numpy import array, empty, float32, float64, ndarray, zeros_like
from numpy.linalg import inv
from pydicom import Dataset

# Dicomnode packages
from dicomnode import dicom
from dicomnode.constants import UNSIGNED_ARRAY_ENCODING, SIGNED_ARRAY_ENCODING
from dicomnode.lib.exceptions import InvalidDataset
from dicomnode import math
from dicomnode.math.types import numpy_image, raw_image_frames
from dicomnode.math.types import MirrorDirection
from dicomnode.math.space import Space, ReferenceSpace



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
    datasets.sort(key=dicom.sort_datasets)

    pivot = datasets[0]
    x_dim = pivot.Columns
    y_dim = pivot.Rows
    z_dim = len(datasets)
    # tags are RescaleIntercept, RescaleSlope
    rescale = (0x00281052 in pivot and 0x00281053 in pivot)

    image_array: numpy_image = empty((z_dim, y_dim, x_dim), dtype=float32)

    for i, dataset in enumerate(datasets):
      image = dataset.pixel_array
      if rescale:
        image_array[i,:,:] = image.astype(float32) * dataset.RescaleSlope\
              + dataset.RescaleIntercept
      else:
        image_array[i,:,:] = image.astype(float32)

    return image_array

class Image:
  def __init__(self,
               image_data: numpy_image,
               space: Space,
               minimum_value=0) -> None:
    self._raw = numpy.array(image_data)
    self._space = space
    self._minimum_value = minimum_value

  def __iter__(self):
    for slice_ in self.raw:
      yield slice_

  @property
  def raw(self):
    return self._raw

  @property
  def space(self):
    return self._space

  @property
  def minimum_value(self):
    return self._minimum_value

  @property
  def shape(self):
    return self.raw.shape

  def __getitem__(self, idx):
    return self.raw[idx]

  def mirror_perspective(self, mirror_direction: MirrorDirection):
    self._raw = math.mirror(self.raw, mirror_direction) # type: ignore
    self.space.mirror_perspective(mirror_direction)

  def transform_to_ras(self):
    """Changes the image such that it's in the RAS reference space
    """

    # This space means that the patient lies like this:

    # Y -------> X
    # |   \  /
    # |    \/
    # |    |
    # |    |
    # |  ------
    # |    o
    # v
    # Z

    # This function is done in a couple of steps:
    # 1. First ensure that the image is in the correct rotation
    # 2. Performs mirroring until we are in RAS space

    if not self.space.is_correct_rotation:
      pass

    reference_space = ReferenceSpace.from_space(self.space)

    match reference_space:
      case ReferenceSpace.RAS:
        pass # Yay
      case ReferenceSpace.RAI:
        self.mirror_perspective(MirrorDirection.Z)
      case ReferenceSpace.RPS:
        self.mirror_perspective(MirrorDirection.Y)
      case ReferenceSpace.RPI:
        self.mirror_perspective(MirrorDirection.YZ)
      case ReferenceSpace.LAS:
        self.mirror_perspective(MirrorDirection.X)
      case ReferenceSpace.LAI:
        self.mirror_perspective(MirrorDirection.XZ)
      case ReferenceSpace.LPS:
        self.mirror_perspective(MirrorDirection.XY)
      case ReferenceSpace.LPI:
        self.mirror_perspective(MirrorDirection.XYZ)
      case None:
        raise Exception("")




  @classmethod
  def from_datasets(cls, datasets: List[Dataset]):
    image_data = build_image_from_datasets(datasets)
    affine = Space.from_datasets(datasets)

    return cls(image_data, affine)

  def __str__(self) -> str:
    return f"An Image over the space:\n{self.space}"

  def __repr__(self) -> str:
    return str(self)


class FramedImage():
  @property
  def space(self):
    return self._space

  @property
  def shape(self):
    return self.raw.shape

  def frame(self, frame: int) -> Image:
    return Image(self.raw[frame], self.space)

  def __init__(self, frames: raw_image_frames, space: Space) -> None:
    self.raw = frames
    self._space = space

  def __iter__(self):
    for frame in self.raw:
      for slice_ in frame:
        yield slice_