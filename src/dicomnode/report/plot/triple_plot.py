# Python Standard Library
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union, Tuple, Dict

# Third party packages
import numpy
import nibabel
from matplotlib.transforms import Affine2D
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import LinearLocator, FuncFormatter

# Dicomnode packages
from dicomnode.lib.logging import get_logger
from dicomnode.report.base_classes import Selector
from dicomnode.report.plot import Plot, rotate_image_90
from dicomnode.report.plot.anatomical_plot import AnatomicalPlot
from dicomnode.report.plot.selector import PercentageSelector

class TriplePlot(Plot):
  """This is a plot of three images next to each other"""

  @dataclass
  class Options:
    planes = (Plot.AnatomicalPlane.SAGITTAL, Plot.AnatomicalPlane.CORONAL, Plot.AnatomicalPlane.TRANSVERSE)
    selector: Union[Selector,
                    Tuple[Selector, Selector, Selector]]
    title: Optional[str] = None
    file_path: Union[Path, str, None] = None
    transform = rotate_image_90

  def __init__(self,
               nifti_image: nibabel.nifti1.Nifti1Image,
               options = Options()) -> None:
    logger = get_logger()
    super().__init__()
    self._figure.set_figheight(6.0)
    self._figure.set_figwidth(15.0)

    plane_fig_1, plane_fig_2, plane_fig_3 = options.planes

    if options.selector 

    figure_options_1 = AnatomicalPlot.Options()
    figure_options_2 = AnatomicalPlot.Options()
    figure_options_3 = AnatomicalPlot.Options()


    figure_1 = AnatomicalPlot(nifti_image, plane_fig_1)
    figure_2 = AnatomicalPlot(nifti_image, plane_fig_2)
    figure_3 = AnatomicalPlot(nifti_image, plane_fig_3)

    self.plot_1 = self._figure.add_subplot(figure_1._axes)
    self.plot_2 = self._figure.add_subplot(figure_2._axes)
    self.plot_3 = self._figure.add_subplot(figure_3._axes)
