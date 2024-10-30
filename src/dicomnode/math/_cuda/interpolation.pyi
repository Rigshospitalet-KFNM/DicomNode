from numpy import ndarray

from dicomnode.math.image import Image
from dicomnode.math.space import Space

def linear(source: Image, target: Space) -> ndarray: ...