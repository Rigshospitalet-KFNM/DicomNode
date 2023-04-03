"""This module have:
  - DicomFactory
  - Grinders

 that depends on nifty.

Nifty is a different image format that is oriented towards data processing rather than storage

This module have extra dependencies on nibabel and dicom2nifty and can be install using pip install dicomnode[nifty]

"""

# Python standard library
import logging
from pathlib import Path
from pprint import pprint
from typing import Iterable, List, Optional

# Third party packages
import numpy
from pydicom import Dataset
from nibabel.nifti1 import Nifti1Image
from dicom2nifti.convert_dicom import dicom_array_to_nifti

# Dicomnode
from dicomnode.lib.dicom_factory import SeriesHeader
from dicomnode.lib.exceptions import IncorrectlyConfigured
from dicomnode.lib.grinders import Grinder
from dicomnode.lib.numpy_factory import NumpyFactory
from dicomnode.lib.logging import get_logger

logger = get_logger()

class NiftiGrinder(Grinder):
  INCORRECTLY_CONFIGURED_ERROR_MESSAGE = "To reorient a nifti you need define a valid Path for output_directory"

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

class NiftiFactory(NumpyFactory):
  def build_from_header(self, header: SeriesHeader, image: Nifti1Image) -> List[Dataset]:
    """Builds a dicom serie from a nifti image and a header

    Args:
      header (dicomnode.lib.dicom_factory.SeriesHeader): The container for header information
      image (nibabel.Nifti1Image): Image to be converted into a dicom series.

    Returns:
      List[pydicom.Dataset]: Series of Dicom Datasets containing the input image
    >>>

    """
    # So nifti in all is wisdom they decided to use coulmn major arrays
    # I have no idea why but a conversion is needed
    # I Need to check test this, and i wrote a util function for this.
    numpy_image = image.get_fdata()

    if image.ndim == 3:
      coloumn_major_shape = image.header.get_data_shape() # type: ignore cols, rows, slices, volumes
      row_major_shape = (coloumn_major_shape[2], coloumn_major_shape[1], coloumn_major_shape[0])

      if numpy_image.shape != row_major_shape:
        numpy_image = numpy.ascontiguousarray(numpy_image.T)

      return super().build_from_header(header, numpy_image)
    else:
      raise NotImplemented # pragma: no cover
