"""This module concerns itself with an 'image' which for this module is the raw
data"""

# Python standard library
from logging import getLogger
from functools import reduce
from operator import mul
from typing import Any, List, Literal, Tuple,Sequence, TypeAlias, Union

# Third party Packages
import nibabel
import numpy
from numpy import array, empty, float32, float64, ndarray, zeros_like
from numpy.linalg import inv
from pydicom import Dataset

# Dicomnode packages
import dicomnode
from dicomnode.constants import UNSIGNED_ARRAY_ENCODING, SIGNED_ARRAY_ENCODING,\
  DICOMNODE_LOGGER_NAME
from dicomnode.lib.exceptions import DimensionalityError
from dicomnode import math
from dicomnode.math.types import numpy_image, raw_image_frames
from dicomnode.math.types import MirrorDirection
from dicomnode.math.space import Space, ReferenceSpace, constrain_space



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

def build_image_from_datasets(datasets: List[Dataset]):
  from dicomnode.dicom import detect_4d_image

  if not len(datasets):
    raise ValueError("Cannot Construct an image from an empty list")

  pivot = datasets[0]

  if detect_4d_image(pivot):
    return _build_4d_image_from_datasets(datasets)
  else:
    return _build_3d_image_from_datasets(datasets)

def _build_4d_image_from_datasets(datasets: List[Dataset]):
  from dicomnode.dicom import get_4d_image_dimensionality
  pivot = datasets[0]

  rescale = (0x00281052 in pivot and 0x00281053 in pivot)

  x_dim = pivot.Columns
  y_dim = pivot.Rows
  z_dim = pivot.NumberOfSlices
  t_dim = get_4d_image_dimensionality(pivot)

  image = empty((t_dim, z_dim, y_dim, x_dim), dtype=float32)

  for dataset in datasets:
    index = (dataset.ImageIndex if 0x0054_1330 in dataset else dataset.InstanceNumber) - 1
    index_3d = index % z_dim
    index_4d = index // z_dim

    frame = dataset.pixel_array.astype(float32)

    if rescale:
      image[index_4d,index_3d,:,:] = frame * dataset.RescaleSlope + dataset.RescaleIntercept
    else:
      image[index_4d,index_3d,:,:] = frame

  return image

def _build_3d_image_from_datasets(datasets: List[Dataset]) -> numpy_image:
  """Builds a 3 dimensional image from datasets

  Args:
      datasets (List[Dataset]): A list of datasets

  Returns:
      numpy_image: _description_
  """
  # This import is here to prevent circular imports
  from dicomnode.dicom import sort_datasets
  datasets.sort(key=sort_datasets)

  pivot = datasets[0]
  x_dim = pivot.Columns
  y_dim = pivot.Rows
  z_dim = len(datasets)
  # tags are RescaleIntercept, RescaleSlope
  rescale = (0x00281052 in pivot and 0x00281053 in pivot)

  image_array: numpy_image = empty((z_dim, y_dim, x_dim), dtype=float32)

  for i, dataset in enumerate(datasets):
    image = dataset.pixel_array.astype(float32)
    if rescale:
      image_array[i,:,:] = image * dataset.RescaleSlope + dataset.RescaleIntercept
    else:
      image_array[i,:,:] = image

  return image_array

class Image:
  def __init__(self,
               image_data: numpy_image,
               space: Space,
               minimum_value=0) -> None:
    self._raw = numpy.array(image_data)
    self._space = space
    self._minimum_value = minimum_value

  @classmethod
  def from_array(cls, image: numpy_image):
    space = Space(numpy.eye(3), [0,0,0], image.shape[-3:])

    return cls(image, space)

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

  @property
  def ndim(self):
    return self.raw.ndim

  def frame(self, frame: int) -> 'Image':
    return self.__class__(self.raw[frame], self.space)

  def frames(self):
    yield from self.raw.reshape(-1, *self.raw.shape[-3:])

  def number_frames(self) -> int:
    if self.raw.ndim < 3:
        raise ValueError(f"Image must be at least 3D, got {self.raw.ndim}D")

    return reduce(mul, self.raw.shape[:-3], 1)

  def slices(self):
    yield from self.raw.reshape(-1, *self.raw.shape[-2:])

  def number_slices(self) -> int:
    return reduce(mul, self.raw.shape[:-2], 1)

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
        raise Exception("Space doesn't have valid basis")


  def embed_image(self, start_coord: 'math.Index', embedding: ndarray):
    """Embeds a volume inside of this image.

    Args:
        start_coord (math.Index): _description_
        embedding (ndarray): _description_

    Raises:
        DimensionalityError: _description_
    """
    if embedding.ndim != 3:
      raise DimensionalityError(f"Embedding doesn't have 3 dimensions, but {embedding.ndim}")

    x_dim, y_dim, z_dim = self.shape

    x_embedding_dim, y_embedding_dim, z_embedding_dim = embedding.shape

    end_coord = math.Index(
      min(start_coord.x + x_embedding_dim, x_dim),
      min(start_coord.y + y_embedding_dim, y_dim),
      min(start_coord.z + z_embedding_dim, z_dim)
    )

    d_s_z = slice(start_coord.z, end_coord.z) # Destination_source_z_dimension
    d_s_y = slice(start_coord.y, end_coord.y)
    d_s_x = slice(start_coord.x, end_coord.x)

    s_s_z = slice(0, min(z_embedding_dim, z_dim - start_coord.z))
    s_s_y = slice(0, min(y_embedding_dim, y_dim - start_coord.y))
    s_s_x = slice(0, min(x_embedding_dim, x_dim - start_coord.x))


    self.raw[d_s_z, d_s_y, d_s_x] = embedding[s_s_z, s_s_y, s_s_x]


  @classmethod
  def from_datasets(cls, datasets: List[Dataset]):
    image_data = build_image_from_datasets(datasets)
    affine = Space.from_datasets(datasets)

    return cls(image_data, affine)

  def __str__(self) -> str:
    return f"An Image over the space:\n{self.space}"

  def __repr__(self) -> str:
    return str(self)

def constrain_array(data: ndarray, restraints: Sequence[Tuple[int, int]]) -> ndarray:
  """Limits an array to a region specified by the restraints

  Note:
    This function is designed to work with the bounding_box function from
    dicomnode.math

  Args:
      data (ndarray): The data to be restraint
      restraints (Tuple[Tuple[int, int], ...]): A tuple with restrains, not that
        these are Inclusive, which is not the default for python slice objects.

  Raises:
      ValueError: Raised if there is an incorrect amount of restraints to the dimensionality of the image

  Returns:
      ndarray: A restrained numpy array



  Example:
  >>> constrain_array(numpy.arange(16).reshape((4,4)) + 1, ((1,2), (1,2)))
  array([[ 6,  7],
      [10, 11]])
  """
  if len(restraints) != data.ndim:
    raise ValueError("Length of restraints do not match the number of dimension of the image")

  slices = tuple(slice(min_, max_ + 1) for min_, max_ in reversed(restraints))

  return data[slices]

def constrain(image: Image, restraints: Sequence[Tuple[int, int]]):
  return Image(
    constrain_array(image.raw, restraints),
    constrain_space(image.space, restraints),
    image.minimum_value
  )


def mask_image(image: 'dicomnode.dicom.series.ImageContainerType', mask):
  from dicomnode.dicom.series import extract_image, ImageContainerType

  if not isinstance(mask, numpy.ndarray):
    mask = extract_image(mask).raw

  masked = image.raw * mask

  constrains = math.bounding_box(masked)

  return constrain(Image(masked, image.space), constrains)

DataContainer = Union[
  Image,
  ndarray,
  nibabel.nifti1.Nifti1Image,
  nibabel.nifti2.Nifti2Image,
]

def get_image_data(container:  DataContainer) -> ndarray:
  if isinstance(container, Image):
    return container.raw.astype(float32)
  elif isinstance(container, nibabel.nifti1.Nifti1Image):
    return container.get_fdata(dtype=float32)
  elif isinstance(container, nibabel.nifti2.Nifti2Image):
    return container.get_fdata(dtype=float32)
  else:
    return container.astype(float32)