"""This module contains a number of functions, which all have the same
call signature. Namely: Iterator[pydicom.Dataset].

These function are preprocessing function, in the sense their purpose is to
convert data into another type.

This is especial important since in the dicom format, an image commonly is
separated into multiple files and therefore multiple object instances.

They are called grinders for their similarly to meat grinders.
you pour some unprocessed data, and out come data mushed together.

Meta grinders are functions which produce grinders, which means they should be
called and not just referenced.
"""

__author__ = "Christoffer Vilstrup Jensen"

# Python Standard Library
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Type, Tuple

# Third party packages
import numpy
from nibabel.nifti1 import Nifti1Image
from pydicom import Dataset
from dicom2nifti.convert_dicom import dicom_array_to_nifti

# Dicom node package
from dicomnode.data_structures.image_tree import DicomTree
from dicomnode.dicom.series import DicomSeries
from dicomnode.lib.exceptions import InvalidDataset, IncorrectlyConfigured
from dicomnode.lib.logging import get_logger

logger = get_logger()

class Grinder(ABC):
  """Interface for injection of grinding method
  """

  @abstractmethod
  def __call__(self, image_generator: Iterable[Dataset]) -> Any:
    raise NotImplemented # pragma: no cover


class IdentityGrinder(Grinder):
  def __call__(self, image_generator: Iterable[Dataset]) -> Iterable[Dataset]:
    """This is an identity function. The iterator is not called.

    Args:
      image_generator (Iterable[Dataset]): An iterator of dataset

    Returns:
      Iterable[Dataset]: The same iterator
    """
    return image_generator

class ListGrinder(Grinder):
  def __call__(self,image_generator: Iterable[Dataset]) -> List[Dataset]:
    """Wraps all datasets in a build-in list

    Args:
      image_generator (Iterable[Dataset]): generator object, list will be build for

    Returns:
      List[Dataset]: A list of datasets
    """
    return list(image_generator)

class DicomTreeGrinder(Grinder):
  def __call__(self, image_generator: Iterable[Dataset]) -> DicomTree:
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
      DicomTree: A tree like data structure, for storing datasets
    """
    return DicomTree(image_generator)

class SeriesGrinder(Grinder):
  """A Grinder to convert the input dicom images to a series. This is very
  useful situation where the input is a single series, as it very close to
  combined tag and numpy grinder

  Produces a dicomnode.dicom.series.DicomSeries Object

  """
  def __call__(self, image_generator: Iterable[Dataset]) -> Any:
    return DicomSeries([ds for ds in image_generator])

class TagGrinder(Grinder):
  def __init__(self, tag_list, optional=False) -> None:
    self.optional = optional
    self.tag_list:List[int] = tag_list

  def __call__(self, image_generator: Iterable[Dataset]) -> Dict[int, Any]:
    value_dict = {}
    pivot: Optional[Dataset] = None
    for dataset in image_generator:
      pivot = dataset # assume the tags are shared
      break
    if pivot is None:
      raise ValueError # Your input probably shouldn't validate with 0 images!

    for tag in self.tag_list:
      if tag in pivot:
        value_dict[tag] = pivot[tag].value
      elif not self.optional:
        raise InvalidDataset

    return value_dict


class ManyGrinder(Grinder):
  def __init__(self, *grinders: Grinder) -> None:
    self.grinders = []
    for grinder in grinders:
      self.grinders.append(grinder)

  def __call__(self, image_generator: Iterable[Dataset]):
    return [grinder(image_generator) for grinder in self.grinders]

class NumpyGrinder(Grinder):
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

  def __numpy_monochrome_grinder(self,  datasets: List[Dataset]):
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
      dataType = self.unsigned_array_encoding.get(pivot.BitsAllocated, None)
    else:
      dataType = self.signed_array_encoding.get(pivot.BitsAllocated, None)

    if dataType is None:
      raise InvalidDataset

    image_array: numpy.ndarray = numpy.empty((z_dim, y_dim, x_dim), dtype=dataType)

    for i, dataset in enumerate(datasets):
      image = dataset.pixel_array
      if rescale:
        image = image.astype(numpy.float64) * dataset.RescaleSlope + dataset.RescaleIntercept
      image_array[i,:,:] = image

    return image_array

  def __call__(self, image_generator: Iterable[Dataset]):
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
    datasets: List[Dataset] = [ds for ds in image_generator]
    pivot = datasets[0]

    if 'InstanceNumber' in pivot:
      datasets.sort(key=lambda ds: ds.InstanceNumber)
    else:
      logger.warning("Instance Number not present in dataset, arbitrary ordering of datasets")

    if pivot.SamplesPerPixel == 1:
      return self.__numpy_monochrome_grinder(datasets)
    if pivot.SamplesPerPixel == 3:
      raise NotImplementedError

    if pivot.SamplesPerPixel == 4:
      logger.error("Dataset contains a retired value for Samples Per Pixel, which is not supported")
    else:
      logger.error("Dataset contains a invalid value for Samples Per Pixel")

    raise InvalidDataset()


class NiftiGrinder(Grinder):
  INCORRECTLY_CONFIGURED_ERROR_MESSAGE = "To reorient a nifti you need \
 define a valid Path for output_directory"

  def __init__(self, output_directory: Optional[Path] = None, reorient_nifti: bool=False) -> None:

    if reorient_nifti and output_directory is None:
      logger.error(self.INCORRECTLY_CONFIGURED_ERROR_MESSAGE)
      raise IncorrectlyConfigured
    if output_directory is not None:
      if not output_directory.exists():
        output_directory.mkdir(parents=True, exist_ok=True)

    self.output_directory = output_directory
    self.reorient_nifti = reorient_nifti


  def __call__(self, datasets: Iterable[Dataset]) -> Nifti1Image:
    """
    Creates a nifti dataset from a dicomnode.server.input.AbstractInput Superclass

    Args:
      datasets (Iterable[Dataset]): A Single dicom series to be converted

    Returns:
      Nifti1Image: Nifti image, note that it have NOT been saved to disk
    """

    lists_datasets = [ds for ds in datasets]
    return_dir = dicom_array_to_nifti(
        dicom_list=lists_datasets,
        output_file=self.output_directory,
        reorient_nifti=self.reorient_nifti
      )

    return return_dir['NII'] # Yeah your documentation is wrong, and you should feel real fucking bad

__all__ = [
  'Grinder',
  'IdentityGrinder',
  'ListGrinder',
  'DicomTreeGrinder',
  'TagGrinder',
  'ManyGrinder',
  'NumpyGrinder',
  'NiftiGrinder'
]
