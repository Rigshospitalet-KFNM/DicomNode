# Python Standard Library
from dataclasses import dataclass, field
from logging import getLogger
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

# Third party packages
import nibabel
from pydicom import Dataset
from matplotlib.gridspec import GridSpec

# Dicomnode packages
from dicomnode.constants import DICOMNODE_LOGGER_NAME
from dicomnode.math.image import build_image_from_datasets
from dicomnode.lib.exceptions import MissingNiftiImage
from dicomnode.report.base_classes import Selector
from dicomnode.report.plot import Plot, rotate_image_90
from dicomnode.report.plot.anatomical_plot import AnatomicalPlot
from dicomnode.report.plot.selector import PercentageSelector

class TriplePlot(Plot):
  """This is a plot of three images next to each other"""

  @dataclass
  class Options:
    planes = (Plot.AnatomicalPlane.SAGITTAL, Plot.AnatomicalPlane.CORONAL, Plot.AnatomicalPlane.TRANSVERSE)
    selector: Union[Selector, Tuple[Selector, Selector, Selector]] = PercentageSelector()
    title: Optional[str] = None
    file_path: Union[Path, str, None] = None
    transform: Callable = rotate_image_90

  def __init__(self,
               images: Union[nibabel.nifti1.Nifti1Image, List[Dataset]],
               options = Options()) -> None:
    logger = getLogger(DICOMNODE_LOGGER_NAME)
    super().__init__(file_path=options.file_path)
    self.figure.set_figheight(6.0)
    self.figure.set_figwidth(15.0)

    grid_spec = GridSpec(1,3, wspace=0.0, hspace=0.0)
    self.plot_1 = self._figure.add_subplot(grid_spec[0])
    self.plot_2 = self._figure.add_subplot(grid_spec[1])
    self.plot_3 = self._figure.add_subplot(grid_spec[2])

    if isinstance(images, List):
      image = build_image_from_datasets(images)
    else:
      image = images.get_fdata()

    if image is None: # pragma: no cover
      logger.error("The input image to the triple plot is missing in the nifti file.")
      raise MissingNiftiImage

    plane_fig_1, plane_fig_2, plane_fig_3 = options.planes
    if isinstance(options.selector, Selector):
      self.figure_1_selector = options.selector
      self.figure_2_selector = options.selector
      self.figure_3_selector = options.selector
    else:
      self.figure_1_selector, self.figure_2_selector, self.figure_3_selector = options.selector

    AnatomicalPlot.plot_axes(self.plot_1, image, plane_fig_1, self.figure_1_selector, options.transform)
    AnatomicalPlot.plot_axes(self.plot_2, image, plane_fig_2, self.figure_2_selector, options.transform)
    AnatomicalPlot.plot_axes(self.plot_3, image, plane_fig_3, self.figure_3_selector, options.transform)
