# Python Standard Library
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

# Third party Packages
from matplotlib.ticker import LinearLocator, FuncFormatter
import nibabel
import numpy
from numpy import ndarray

# Dicomnode Packages
from dicomnode.report.plot import Plot, rotate_image_90
from dicomnode.report.plot.selector import Selector, PercentageSelector

class AnatomicalPlot(Plot):
  class Options:
    title: Optional[str]
    selector: Selector = field(default_factory=PercentageSelector)
    file_path: Union[str, Path, None] = None
    transform = rotate_image_90


  def __init__(self, nifti_image: nibabel.nifti1.Nifti1Image, plane: Plot.AnatomicalPlane, options = Options()) -> None:
    self.__options = options
    self.__image = nifti_image
    self.__plane = plane
    super().__init__()
    self._axes = self.figure.add_axes((5,5,5,5))
    self._axes.set_axis_off()
    image_data = self.__image.get_fdata()
    image, image_index = options.selector(Plot.PlaneImages(image_data, self.__plane))

    transformed_image = self.__options.transform(image)

    image_show_key_word_args = {
      "cmap" : "gray",
      "origin" : "lower",
      "aspect" : "equal",
    }

    self._axes.imshow(transformed_image, **image_show_key_word_args)


