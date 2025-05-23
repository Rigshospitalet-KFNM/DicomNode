# This is a interface file for the pybind11 library created from:
# low_level_src/python_entry_point.cu

from typing import Any, List, Tuple
from numpy import ndarray

class CudaError:
  def __bool__(self): ...

  def __int__(self): ...

class DicomnodeError:
  def __bool__(self): ...

  def __int__(self): ...

class DeviceProperties: ...

# low_level_src/python/python_mirror.cu
def mirror_x(arr: ndarray) -> ndarray: ...
def mirror_y(arr: ndarray) -> ndarray: ...
def mirror_z(arr: ndarray) -> ndarray: ...
def mirror_xy(arr: ndarray) -> ndarray: ...
def mirror_xz(arr: ndarray) -> ndarray: ...
def mirror_yz(arr: ndarray) -> ndarray: ...
def mirror_xyz(arr: ndarray) -> ndarray: ...

# low_level_src/python/python_bounding_box.cu
def bounding_box(arr: ndarray) -> Tuple[CudaError, List[int]]: ...


def print_device_image(image) -> Any: ...


def get_device_properties() -> Tuple[CudaError, DeviceProperties]: ...

from . import interpolation
from . import labeling

all = [
  interpolation,
  labeling,
  get_device_properties,
  print_device_image,
  bounding_box,
  mirror_x,
  mirror_y,
  mirror_z,
  mirror_xy,
  mirror_xz,
  mirror_yz,
  mirror_xyz
]