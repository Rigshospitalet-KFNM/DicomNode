from typing import Tuple

from numpy import ndarray

from dicomnode.math.image import Image
from dicomnode.math.space import Space

from dicomnode.math._cuda import DicomnodeError

def linear(source: Image, target: Space) -> Tuple[DicomnodeError, ndarray]: ...