# This is a interface file for the pybind11 library created from:
# cuda_src/python_entry_point.cu

from typing import Any, List, Tuple
from numpy import ndarray

from dicomnode.math.image import Image

class DicomnodeCudaError:
  def __bool__(self): ...

  def __int__(self): ...

class DicomnodeError:
  def __init__(self, error_code) -> None: ...

  def __bool__(self): ...

  def __int__(self): ...

class DicomnodeDeviceProperties: ...

# cuda_src/python/python_mirror.cu
def mirror_x(arr: ndarray) -> ndarray: ...
def mirror_y(arr: ndarray) -> ndarray: ...
def mirror_z(arr: ndarray) -> ndarray: ...
def mirror_xy(arr: ndarray) -> ndarray: ...
def mirror_xz(arr: ndarray) -> ndarray: ...
def mirror_yz(arr: ndarray) -> ndarray: ...
def mirror_xyz(arr: ndarray) -> ndarray: ...

# cuda_src/python/python_bounding_box.cu
def bounding_box(arr: ndarray) -> Tuple[DicomnodeCudaError, List[int]]: ...

# cuda_src/python/python_center_of_gravity.cu
def center_of_gravity(image: ndarray) -> Tuple[DicomnodeCudaError, Tuple[float,float,float]]: ...

def get_device_properties() -> Tuple[DicomnodeCudaError, DicomnodeDeviceProperties]: ...


from . import interpolation
from . import labeling
from . import registration

all = [
  interpolation,
  labeling,
  registration,
  get_device_properties,
  center_of_gravity,
  bounding_box,
  mirror_x,
  mirror_y,
  mirror_z,
  mirror_xy,
  mirror_xz,
  mirror_yz,
  mirror_xyz
]