# Python Standard Library
from collections.abc import Iterable

# Third party modules
import numpy

class Extent:
  """This class represents a domain or an extent in a linear space"""

  def __init__(self, *args) -> None:
    if len(args) == 0:
      raise ValueError("You need at least one argument to construct an Extent")

    first_arg = args[0]

    if isinstance(first_arg, Iterable) and len(args) == 1:
      self.extent = numpy.array(first_arg)
    elif isinstance(first_arg, Iterable):
      raise ValueError("Either pass an iterable object or some number of numbers")
    else:
      self.extent = numpy.array([ a for a in args], dtype=numpy.uint32)

  def __array__(self, dtype=None):
    if dtype is not None:
      return self.extent.astype(dtype)
    return self.extent

  def __getitem__(self, key):
    return self.extent[key]

  def __eq__(self, value: object) -> bool:
    if isinstance(value, Iterable):
      try:
        return all(x == y for x,y in zip(value, self.extent, strict=True))
      except ValueError:
        return False
    raise TypeError(f"Unable to compare Extent to {value}")

  def __iter__(self):
    yield from self.extent

  @property
  def x(self):
    return self.extent[-1]

  @property
  def y(self):
    return self.extent[-2]

  @property
  def z(self):
    return self.extent[-3]
