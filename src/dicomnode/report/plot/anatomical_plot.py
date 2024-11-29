# Python Standard Library
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Union

# Third party Packages
from matplotlib.axes import Axes
from matplotlib.ticker import LinearLocator, FuncFormatter
import nibabel
import numpy
from numpy import ndarray

# Dicomnode Packages
from dicomnode.report.plot import Plot, rotate_image_90
from dicomnode.report.plot.selector import Selector, PercentageSelector

class AnatomicalPlot(Plot):
  @dataclass
  class Options:
    title: Optional[str] = None
    selector: Selector = field(default_factory=PercentageSelector)
    file_path: Union[str, Path, None] = None
    transform: Callable = rotate_image_90

  @classmethod
  def plot_axes(cls, axes: Axes, images: ndarray, plane: Plot.AnatomicalPlane, selector: Selector, transform: Callable):
    axes.set_axis_off()
    plane_images = Plot.PlaneImages(images, plane)
    image, index = selector(plane_images)
    transformed_image = transform(image)
    image_show_key_word_args = {
      "cmap" : "gray",
      "origin" : "lower",
      "aspect" : "equal",
    }

    axes.imshow(transformed_image, **image_show_key_word_args) # type: ignore # this is why enum are better

  def __init__(self, nifti_image: nibabel.nifti1.Nifti1Image, plane: Plot.AnatomicalPlane, options = Options()) -> None:
    self.__options = options
    self.__image = nifti_image
    self.__plane = plane
    super().__init__()
    image_data = self.__image.get_fdata()
    if image_data is None:
      raise Exception
    self._axes = self.figure.add_axes((5,5,5,5))
    self.plot_axes(self.figure.axes[0], image_data, plane, options.selector, options.transform)
