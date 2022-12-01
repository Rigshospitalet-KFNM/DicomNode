"""This module contains a number of functions, which all have the same
call signature. Namely: Iterator[pydicom.Dataset].

They are called grinders for their similarly to meat grinders.
you pour some unprocessed data, and out come data mushed together.

Meta grinders are functions which produce grinders, which means they should be
called and not just referenced.
"""

__author__ = "Christoffer Vilstrup Jensen"

from typing import Iterator, List, Callable, Any
from pydicom import Dataset

from dicomnode.lib.imageTree import DicomTree

def identity_grinder(image_generator: Iterator[Dataset] ) -> Iterator[Dataset]:
  """This is an identity function. The iterator is not called.

  Args:
      image_generator (Iterator[Dataset]): An iterator of dataset

  Returns:
      Iterator[Dataset]: The same iterator
  """
  return image_generator

def list_grinder(image_generator: Iterator[Dataset]) -> List[Dataset]:
  return list(image_generator)

def dicom_tree_grinder(image_generator: Iterator[Dataset]) -> DicomTree:
  return DicomTree(image_generator)

def many_meta_grinder(*grinders: Callable[[Iterator[Dataset]], Any]) -> Callable[[Iterator[Dataset]], List[Any]]:
  """This meta grinder combines any number of grinders

  Args:
    grinders (Callable[[Iterator[Dataset]], Any])

  Returns:
      Callable[[Iterator[Dataset]], List[Any]]: _description_
  """
  def retFunc(image_generator: Iterator[Dataset]) -> List[Any]:
    grinded: List[Any] = []
    for grinder in grinders:
      grinded.append(grinder(image_generator))
    return grinded
  return retFunc

try:
  import numpy
  def numpy_grinder(datasets: Iterator[Dataset]) -> numpy.ndarray:
    datasets: List[Dataset] = list(datasets)
    pivot = datasets[0]
    x_dim = pivot.Columns
    y_dim = pivot.Rows
    z_dim = len(datasets)

    image_array: numpy.ndarray = numpy.empty((x_dim, y_dim, z_dim))

    for i, dataset in enumerate(datasets):
      image_array[i,:,:] = dataset.pixel_array

    return image_array

except ImportError:
  pass
