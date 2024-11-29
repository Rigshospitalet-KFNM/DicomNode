from typing import Tuple

from numpy import ndarray

from dicomnode.math.image import Image
from dicomnode.math.space import Space

from dicomnode.math._cuda import DicomnodeError

def linear(source: Image, target: Space) -> Tuple[DicomnodeError, ndarray]:
  """Creates a linear interpolation over the source of the target space

  Source is found at low_level_src/python_interpolation.cu

  Args:
    source (Image): The Image that is the source of the interpolation
    target (Space): The space of the image to be created

  Returns:
    Tuple(DicomnodeError, ndarray) :

  """
  ...