"""Affine Matrixes is describe a the pixel space of an image


"""

# Python standard library
from enum import Enum
from typing import List, Literal, Tuple, TypeAlias

# Third party packages
from nibabel.nifti1 import Nifti1Image
import numpy
from numpy import array, absolute, dtype, float32, identity, ndarray, uint32
from numpy.linalg import inv
from pydicom import Dataset

# Dicomnode packages
from dicomnode.lib.exceptions import NonReducedBasis
from dicomnode.math.types import MirrorDirection, ROTATION_MATRIX_90_DEG_X,\
  ROTATION_MATRIX_90_DEG_Y, ROTATION_MATRIX_90_DEG_Z

RawBasisMatrix: TypeAlias = ndarray[Tuple[int, ...], dtype[float32]]

def is_affine_in_correct_rotation(affine):
  abs_affine = absolute(affine)

  return abs_affine[0,1] < abs_affine[0,0] and \
         abs_affine[0,2] < abs_affine[0,0] and \
         abs_affine[1,0] < abs_affine[1,1] and \
         abs_affine[1,2] < abs_affine[1,1] and \
         abs_affine[2,0] < abs_affine[2,2] and \
         abs_affine[2,1] < abs_affine[2,2]

def is_positive(num):
  return 0 < num


class Space:
  """A Space is a 3 dimensional linear space with a starting point and a domain,
  which represent the coordinate system used describes an image.

  It contains relevant information:
  * basis - 3x3 basis, note that the vector are not unit vectors
  * starting_point - The point in space for index (0,0,0)
  * extent - The values z,y,x such that all indexes (k,j,i) satisfying:
    0 < i < x, 0 < j < y, 0 < k < z
    has a image value and all indexes not satisfying this condition do not.

  This class doesn't cover images, with overlapping slices or non uniform slice
  thickness, which you might encounter.
  """
  @property
  def reference_space(self):
    return ReferenceSpace.from_space(self)

  @property
  def basis(self):
    return self._basis

  @basis.setter
  def basis(self, basis):
    self._basis = basis
    self._inverted_basis = inv(basis)

  @property
  def inverted_basis(self):
    return self._inverted_basis

  @property
  def extent(self):
    return self._extent

  @property
  def starting_point(self):
    return self._starting_point

  def __init__(self, basis, starting_point, domain):
    """A Space is a 3 dimensional linear space with a starting point and a domain,
  which represent the coordinate system used describes an image.

  It contains relevant information:
  * basis - 3x3 basis, note that the vector are not unit vectors
  * starting_point - The point in space for index (0,0,0)
  * extent - The values z,y,x such that all indexes (k,j,i) satisfying:
    0 < i < x, 0 < j < y, 0 < k < z
    has a image value and all indexes not satisfying this condition do not.

  This class doesn't cover images, with overlapping slices or non uniform slice
  thickness, which you might encounter.

    Args:
      basis : The basis of the space A object convertible to a 3 x 3 numpy matrix.
      starting_point. : The starting point of the space - An object convertible to a numpy array of shape(3,)
      extent : z,y,x of the space
  """
    self._basis = numpy.array(basis, dtype=float32)
    self._inverted_basis = numpy.array(inv(self._basis),dtype=float32)
    self._starting_point = numpy.array(starting_point, dtype=float32)
    self._extent = numpy.array(domain, dtype=uint32)

  def __eq__(self, other):
    # This is mostly useful for testing
    if not isinstance(other, Space):
      raise TypeError(f"Compared {other.__class__.__name__} with a space")

    return (self.basis == other.basis).all() and\
           (self.starting_point == other.starting_point).all() and\
           (self.extent == other.extent).all()

  def coords(self):
    index = 0
    points = numpy.prod(self.extent)

    while index < points:
      x = index % self.extent[2]
      y = (index // self.extent[2]) % self.extent[1]
      z = index // (self.extent[1] * self.extent[2])

      yield numpy.array([x,y,z])

      index += 1

  @classmethod
  def from_nifti(cls, nifti: Nifti1Image):
    maybe_affine = nifti.affine
    data = nifti.get_fdata()

    if maybe_affine is not None:
      affine = maybe_affine[:3, :3]
      start_point = maybe_affine[:3, 3]
    else:
      affine = identity(3, dtype=float32)
      start_point = [0,0,0]

    # Yeah this sucks, but most likely the data is already available
    if data.flags.f_contiguous:
      shape = numpy.array([s for s in reversed(data.shape)])[:3]
    else:
      shape = numpy.array(data.shape)[:3]

    return cls(affine, start_point, shape)

  def to_affine(self):
    affine = numpy.eye(4)

    affine[:3,:3] = self.basis
    affine[3, :3] = self.starting_point

    return affine

  @classmethod
  def from_datasets(cls, datasets: List[Dataset]):
    first_dataset = datasets[0]
    slices_per_image = first_dataset.NumberOfSlices if 'NumberOfSlices' in first_dataset else len(datasets)
    slice_coordinates = numpy.array(first_dataset.ImagePositionPatient, dtype=float32)
    image_orientation = first_dataset.ImageOrientationPatient

    thickness_x = first_dataset.PixelSpacing[0]
    thickness_y = first_dataset.PixelSpacing[1]
    thickness_z = first_dataset.SliceThickness

    affine_raw = numpy.array([
      [thickness_x * image_orientation[0], thickness_y * image_orientation[3], 0],
      [thickness_x * image_orientation[1], thickness_y * image_orientation[4], 0],
      [thickness_x * image_orientation[2], thickness_y * image_orientation[5], thickness_z],
    ], dtype=float32)

    slice_number = (first_dataset.InstanceNumber - 1) % first_dataset.NumberOfSlices if 'NumberOfSlices' in first_dataset else first_dataset.InstanceNumber - 1

    starting_coordinates = (slice_number * numpy.array([0,0,-1])) @ affine_raw + slice_coordinates

    return cls(
      affine_raw,
      starting_coordinates ,
      (slices_per_image, first_dataset.Columns, first_dataset.Rows)
    )

  def _mirror_x_direction(self):
    """THIS FUNCTION LEADS THE CLASS IN AN INVALID STATE, UNLESS YOU HAVE CALLED
    self._inverted_basis = inv(self.basis)
    """
    self._starting_point = self.starting_point + self.basis[0,:] * self.extent[2]
    self._basis[0,:] = -self.basis[0,:]

  def _mirror_y_direction(self):
    """THIS FUNCTION LEADS THE CLASS IN AN INVALID STATE, UNLESS YOU HAVE CALLED
    self._inverted_basis = inv(self.basis)
    """
    self._starting_point = self.starting_point + self.basis[1,:] * self.extent[1]
    self._basis[1,:] = -self.basis[1,:]

  def _mirror_z_direction(self):
    """THIS FUNCTION LEADS THE CLASS IN AN INVALID STATE, UNLESS YOU HAVE CALLED
    self._inverted_basis = inv(self.basis)
    """
    self._starting_point = self.starting_point + self.basis[2,:] * self.extent[0]
    self._basis[2,:] = -self.basis[2,:]


  def mirror_perspective(self, direction: MirrorDirection):
    """Changes the starting point and invert the basis vector matching the input
    direction. If the function spans a space, this function does update that
    space.


    Visualizing an 1D example (Looks best in real docs):
      |-------------------------------------|
      ^                                     ^
      startingPoint                         starting point + basis * self.extent

    After mirroring:
      |-------------------------------------|
      ^                                     ^
      startingPoint + basis * self.extent   starting point


    """
    match direction:
      case MirrorDirection.X:
        self._mirror_x_direction()
      case MirrorDirection.XY:
        self._mirror_x_direction()
        self._mirror_y_direction()
      case MirrorDirection.XZ:
        self._mirror_x_direction()
        self._mirror_z_direction()
      case MirrorDirection.XYZ:
        self._mirror_x_direction()
        self._mirror_y_direction()
        self._mirror_z_direction()
      case MirrorDirection.Y:
        self._mirror_y_direction()
      case MirrorDirection.YZ:
        self._mirror_y_direction()
        self._mirror_z_direction()
      case MirrorDirection.Z:
        self._mirror_z_direction()

    self._inverted_basis = inv(self.basis)

  def __str__(self):
    return (f"Space over extend x: {self.extent[2]}, y: {self.extent[1]} z: {self.extent[0]}\n"
            f"Starting point at ({self.starting_point[0]},{self.starting_point[1]}, {self.starting_point[2]})\n"
            f"Basis:\n"
            f"{self.basis[0,0]} {self.basis[0,1]} {self.basis[0,2]}\n"
            f"{self.basis[1,0]} {self.basis[1,1]} {self.basis[1,2]}\n"
            f"{self.basis[2,0]} {self.basis[2,1]} {self.basis[2,2]}")

  def __repr__(self) -> str:
    return str(self)

  @property
  def is_correct_rotation(self) -> bool:
    """Detect if the image is rotated such that the reference space is not None

    Returns:
        bool: return True if there's a reference space that matches
    """
    return is_affine_in_correct_rotation(self.basis)


  def __dominant_axis(self, vector):
    abs_vec = absolute(vector)
    if abs_vec[0] < abs_vec[1]:
      if abs_vec[1] < abs_vec[2]:
        return 2
      else:
        return 1
    else:
      if abs_vec[0] < abs_vec[2]:
        return 2
      else:
        return 0

  def get_rotation_matrix_to_standard_space(self) -> ndarray:
    axis_x_dom = self.__dominant_axis(self.basis[0])
    axis_y_dom = self.__dominant_axis(self.basis[1])
    axis_z_dom = self.__dominant_axis(self.basis[2])

    if axis_x_dom == axis_y_dom or axis_x_dom == axis_z_dom or axis_y_dom == axis_z_dom:
      raise NonReducedBasis("The Basis is invalid")

    if axis_x_dom == 0:
      if axis_y_dom == 1:
        return identity(3)
      else:
        return ROTATION_MATRIX_90_DEG_X
    elif axis_x_dom == 1:
      if axis_y_dom == 0:
        return ROTATION_MATRIX_90_DEG_Z
      else:
        return ROTATION_MATRIX_90_DEG_Z @ ROTATION_MATRIX_90_DEG_X
    else: # axis_x_dom == 2
      if axis_y_dom == 0:
        return ROTATION_MATRIX_90_DEG_X @ ROTATION_MATRIX_90_DEG_Z
      else:
        return ROTATION_MATRIX_90_DEG_Y

  def _rotate_extent(self, rotation_matrix):
    new_extent = numpy.empty((3,), dtype=uint32)

    new_extent[0] = self.extent[self.__dominant_axis(rotation_matrix[0])]
    new_extent[1] = self.extent[self.__dominant_axis(rotation_matrix[1])]
    new_extent[2] = self.extent[self.__dominant_axis(rotation_matrix[2])]

    return new_extent

  def _rotate_starting_point(self, rotation_matrix):
    new_starting_point = numpy.copy(self.starting_point)

    for i in range(3):
      index = self.__dominant_axis(rotation_matrix[i])
      rotation_direction = rotation_matrix[i][index]

      if rotation_direction == 1:
        pass
      elif rotation_direction == -1:
        new_starting_point += (self.extent[i] * self.basis[:,i])

      else:
        raise Exception("")


    return new_starting_point

  def rotate(self, rotation_matrix):
    new_basis = self.basis @ rotation_matrix
    new_starting_point = self._rotate_starting_point(rotation_matrix)
    new_extent = self._rotate_extent(rotation_matrix)

    return Space(
      new_basis,
      new_starting_point,
      new_extent
    )

class ReferenceSpace(Enum):
  """These are the possible reference spaces a study can be in.

  Here we use the terms defined in:
  https://nipy.org/nibabel/coordinate_systems.html
  """

  RAS = 0
  """Indicate that the image have an affine matrix on the form:
  | x, 0, 0 |
  | 0, y, 0 |
  | 0, 0, z |
  """

  RAI = 1
  """Indicate that the image have an affine matrix on the form:
  | x, 0,  0 |
  | 0, y,  0 |
  | 0, 0, -z |
  """
  RPS = 2
  """Indicate that the image have an affine matrix on the form:
  | x,  0, 0 |
  | 0, -y, 0 |
  | 0,  0, z |
  """
  RPI = 3
  """Indicate that the image have an affine matrix on the form:
  | x, 0,  0 |
  | 0, -y, 0 |
  | 0, 0, -z |
  """
  LAS = 4
  """Indicate that the image have an affine matrix on the form:
  | -x, 0, 0 |
  | 0, y, 0 |
  | 0, 0, z |
  """
  LAI = 5
  """Indicate that the image have an affine matrix on the form:
  | -x, 0, 0 |
  | 0, y, 0 |
  | 0, 0, -z |
  """
  LPS = 6
  """Indicate that the image have an affine matrix on the form:
  | -x, 0, 0 |
  | 0, -y, 0 |
  | 0, 0, z |
  """
  LPI = 7
  """Indicate that the image have an affine matrix on the form:
  | -x, 0, 0 |
  | 0, -y, 0 |
  | 0, 0, -z |
  """

  @classmethod
  def from_space(cls, space: Space):
    if not space.is_correct_rotation:
      return None

    return cls.from_affine(space.basis)

  @classmethod
  def from_affine(cls, affine: ndarray):
    x_coord = affine[0,0]
    y_coord = affine[1,1]
    z_coord = affine[2,2]


    if is_positive(x_coord) and is_positive(y_coord) and is_positive(z_coord):
      return cls.RAS
    if is_positive(x_coord) and is_positive(y_coord) and not is_positive(z_coord):
      return cls.RAI
    if is_positive(x_coord) and not is_positive(y_coord) and is_positive(z_coord):
      return cls.RPS
    if is_positive(x_coord) and not is_positive(y_coord) and not is_positive(z_coord):
      return cls.RPI
    if not is_positive(x_coord) and is_positive(y_coord) and is_positive(z_coord):
      return cls.LAS
    if not is_positive(x_coord) and is_positive(y_coord) and not is_positive(z_coord):
      return cls.LAI
    if not is_positive(x_coord) and not is_positive(y_coord) and is_positive(z_coord):
      return cls.LPS
    if not is_positive(x_coord) and not is_positive(y_coord) and not is_positive(z_coord):
      return cls.LPI

    return None
