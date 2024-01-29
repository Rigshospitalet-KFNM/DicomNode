# Python Standard Library
from typing import Sequence, Union

# Third party Packages
import numpy

# Dicomnode packages
from dicomnode.report.base_classes import Selector

class PercentageSelector(Selector):
  def __init__(self, percentage: Union[float, int] = 50):
    super().__init__()
    if isinstance(percentage, int):
      self.percentage = percentage / 100
    else:
      self.percentage = percentage

  def __call__(self, images: Sequence[numpy.ndarray]):
    middle = int(len(images) * self.percentage)
    return images[middle], middle

class AverageSelector(Selector):
  def __call__(self, images: Sequence[numpy.ndarray]):
    pivot = None
    pivot_index = -1
    pivot_average = -numpy.Infinity

    for index, image in enumerate(images):
      image_average = numpy.mean(image)
      if pivot_average < image_average: # type: ignore # Yeah this warning is fucking stupid
        pivot = image
        pivot_index = index
        pivot_average = image_average

    return pivot, pivot_index

class MaxSelector(Selector):
  def __call__(self, images: Sequence[numpy.ndarray]):
    pivot = None
    pivot_index = -1
    pivot_max = -numpy.Infinity

    for index, image in enumerate(images):
      image_max = image.max()
      if pivot_max < image_max:
        pivot = image
        pivot_index = index
        pivot_max = image_max

    return pivot, pivot_index

