from typing import Any, Tuple

from numpy import ndarray

from dicomnode.math.image import Image
from dicomnode.math.space import Space

def linear(source_image: Image, target_space: Space) -> Tuple[Any, ndarray]: ...
