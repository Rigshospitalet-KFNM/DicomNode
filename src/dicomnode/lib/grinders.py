"""This module contains a number of functions, which all have the same
call signature. Namely: Iterator[pydicom.Dataset].

They are called grinders for their similarly to meat grinders.
you pour some unprocessed data, and out come data mushed together.

Meta grinders are functions which produce grinders, which means they should be
called and not just referenced.
"""

__author__ = "Christoffer Vilstrup Jensen"

from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Type, Tuple

from pydicom import Dataset
from dicomnode.lib.exceptions import InvalidDataset
from dicomnode.lib.imageTree import DicomTree

import numpy
import logging

logger = logging.getLogger("dicomnode")


def identity_grinder(image_generator: Iterable[Dataset] ) -> Iterable[Dataset]:
  """This is an identity function. The iterator is not called.

  Args:
      image_generator (Iterable[Dataset]): An iterator of dataset

  Returns:
      Iterable[Dataset]: The same iterator
  """
  return image_generator

def list_grinder(image_generator: Iterable[Dataset]) -> List[Dataset]:
  """Wraps all datasets in a build-in list

  Args:
      image_generator (Iterable[Dataset]): generator object, list will be build for

  Returns:
      List[Dataset]: A list of datasets
  """
  return list(image_generator)

def dicom_tree_grinder(image_generator: Iterable[Dataset]) -> DicomTree:
  """Constructs a DicomTree from the input

  Requires That each Dataset have the tags:
    PatientID
    SOPInstanceUID
    StudyInstanceUID
    SeriesInstanceUID

  Additional functionality is available if the tags are present:
    StudyDescription - Names the Study trees
    SeriesDescription - Names the Series trees
    PatientName - Names the Patient Trees

  Args:
      image_generator (Iterable[Dataset]): generator object, the tree will be build from

  Returns:
      DicomTree: A datastructure
  """
  return DicomTree(image_generator)

def tag_meta_grinder(tag_list: List[int], optional=False) -> Callable[[Iterable[Dataset]], List[Tuple[int, Any]]]:
  """Generates a function that extracts values at a tag
      The tags are taken from an arbitrary dataset from the collection of datasets
      In other words ensure that the tag is equal among all datasets of the collection

  Args:
      tag_list (List[int]): The list of tags to be extracted
      optional (bool, optional): if False causes an exception if a tag is missing in the dataset. Defaults to False.

  Returns:
      Callable[[Iterable[Dataset]], List[Tuple[int, Any]]]: function which does the extraction.

  Example:
    >>>grinder = tag_meta_grinder([0x00100010])
    >>>dataset = pydicom.Dataset()
    >>>dataset.PatientName = "patient_name"
    >>>grinder([dataset])
    [(0x00100010, "patient_name")]
  """
  def ret_func(image_generator: Iterable[Dataset]) -> List[Tuple[int, Any]]:
    value_list: List[Tuple[int, Any]] = []
    pivot: Optional[Dataset] = None
    for dataset in image_generator:
      pivot = dataset # assume the tags are shared
      break
    if pivot is None:
      raise ValueError # Your input probably shouldn't validate with 0 images!

    for tag in tag_list:
      if tag in pivot:
        value_list.append((tag, pivot[tag].value))
      elif not optional:
        raise InvalidDataset

    return value_list
  return ret_func


def many_meta_grinder(*grinders: Callable[[Iterable[Dataset]], Any]) -> Callable[[Iterable[Dataset]], List[Any]]:
  """This meta grinder combines any number of grinders

  Args:
    grinders (Callable[[Iterator[Dataset]], Any])

  Returns:
      Callable[[Iterator[Dataset]], List[Any]]: _description_
  """
  def retFunc(image_generator: Iterable[Dataset]) -> List[Any]:
    grinded: List[Any] = []
    for grinder in grinders:
      grinded.append(grinder(image_generator))
    return grinded
  return retFunc

unsigned_array_encoding: Dict[int, Type[numpy.unsignedinteger]] = {
  8 : numpy.uint8,
  16 : numpy.uint16,
  32 : numpy.uint32,
  64 : numpy.uint64,
}

signed_array_encoding: Dict[int, Type[numpy.signedinteger]] = {
  8 : numpy.int8,
  16 : numpy.int16,
  32 : numpy.int32,
  64 : numpy.int64,
}


def _numpy_monochrome_grinder(datasets: List[Dataset]) -> numpy.ndarray:
  pivot = datasets[0]
  x_dim = pivot.Columns
  y_dim = pivot.Rows
  z_dim = len(datasets)
  rescale = (0x00281052 in pivot and 0x00281053 in pivot)

  if 0x7FE00008 in pivot:
    dataType = numpy.float32
  elif 0x7FE00009 in pivot:
    dataType = numpy.float64
  elif rescale:
    dataType = numpy.float64
  elif pivot.PixelRepresentation == 0:
    dataType = unsigned_array_encoding.get(pivot.BitsAllocated, None)
  else:
    dataType = signed_array_encoding.get(pivot.BitsAllocated, None)

  if dataType is None:
    raise InvalidDataset

  image_array: numpy.ndarray = numpy.empty((z_dim, y_dim, x_dim), dtype=dataType)

  for i, dataset in enumerate(datasets):
    if rescale:
      image = (numpy.asarray(dataset.pixel_array, dtype=numpy.float64) - dataset.RescaleIntercept) * dataset.RescaleSlope
    else:
      image = dataset.pixel_array
    image_array[i,:,:] = image

  return image_array


def numpy_grinder(datasets_iterator: Iterable[Dataset]) -> numpy.ndarray:
  """Constructs a 3d volume from a collections of pydicom.Dataset

    Args:
      datasets_iterator: Iterable[Dataset]

    Each dataset Requires Tags:
      0x00280002 - SamplesPerPixel, with value 1 or 3
      0x00280004 - PhotometricInterpretation
      0x00280010 - Rows
      0x00280011 - Columns
      0x00280100 - BitsAllocated
      0x00280103 - PixelRepresentation
      0x7FE00008 or 0x7FE0009 or 0x7FE00010 - Image data

    Each dataset requires values:

    Additional Functionality available if tags are present
      0x00281052 and 0x00281053, RescaleIntercept and RescaleSlope
        rescales the the picture to original values, allows slice based scaling
      0x00200013 InstanceNumber - Sorts the dataset ensuring correct order

  """
  datasets: List[Dataset] = [ds for ds in datasets_iterator]
  pivot = datasets[0]

  if 'InstanceNumber' in pivot:
    datasets.sort(key=lambda ds: ds.InstanceNumber)
  else:
    logger.warn("Instance Number not present in dataset, arbitrary ordering of datasets")

  if pivot.SamplesPerPixel == 1:
    return _numpy_monochrome_grinder(datasets)
  if pivot.SamplesPerPixel == 3:
    raise NotImplementedError

  if pivot.SamplesPerPixel == 4:
    logger.error("Dataset contains a retired value for Samples Per Pixel, which is not supported")
  else:
    logger.error("Dataset contains a invalid value for Samples Per Pixel")

  raise InvalidDataset()