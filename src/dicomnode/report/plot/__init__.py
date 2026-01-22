
"""This module provides a standardized way of producing figures, plots and
various displays for a standardized way to present the reports"""

# Python3 Standard Library
from enum import Enum
from pathlib import Path
from typing import Optional, Sequence
from uuid import uuid4

# Third party packages
import numpy
from numpy import ndarray
from matplotlib.figure import Figure
from matplotlib.pyplot import close as close_figure
from pylatex import StandAloneGraphic, LineBreak

# Dicomnode packages
from dicomnode import library_paths
from dicomnode.report import Report
from dicomnode.report.base_classes import LaTeXComponent

def rotate_image_90(image: ndarray):
  return numpy.rot90(image, 3)

class Plot(LaTeXComponent):
  # Enums
  class AnatomicalPlane(Enum):
    SAGITTAL=0
    CORONAL=1
    TRANSVERSE=2

  class PlaneImages(Sequence):
    def __init__(self, image: numpy.ndarray, plane: 'Plot.AnatomicalPlane') -> None:
      self.image = image
      self.plane = plane

    def __iter__(self):
      for plane in numpy.rollaxis(self.image, self.plane.value):
        yield plane

    def __len__(self):
      return self.image.shape[self.plane.value]

    def __getitem__(self, index):
      if isinstance(index, slice):
        start, stop, step = index.start, index.stop, index.step
        if self.plane == Plot.AnatomicalPlane.SAGITTAL:
          return self.image[start:stop:step, :, :]
        if self.plane == Plot.AnatomicalPlane.CORONAL:
          return self.image[:, start:stop:step, :]
        if self.plane == Plot.AnatomicalPlane.TRANSVERSE:
          return self.image[:, :, start:stop:step]

      if self.plane == Plot.AnatomicalPlane.SAGITTAL:
        return self.image[index, :, :]
      if self.plane == Plot.AnatomicalPlane.CORONAL:
        return self.image[:, index, :]
      if self.plane == Plot.AnatomicalPlane.TRANSVERSE:
        return self.image[:, :, index]

    def __add__(self, other): # required for sequences
      if other.plane == self.plane:
        return self.__class__(self.image + other.plan, self.plane)
      raise Exception

    def __mul__(self, num): # required for sequences
      val = self
      for _ in range(num - 1):
        val = self + val
      return val

  def __init__(self, figure: Optional[Figure] = None, file_path = None) -> None:
    self._file_path = file_path
    if figure is None:
      self._figure = Figure()
    else:
      self._figure = figure

  @property
  def figure(self):
    return self._figure

  def show(self):
    self._figure.show() #pragma: no cover

  @property
  def file_path(self) -> Path:
    if self._file_path is None:
      self._file_path = Path(library_paths.figure_directory / (self.__class__.__name__ + str(uuid4()) + '.png'))
    elif not isinstance(self._file_path, Path):
      self._file_path = Path(self._file_path)
    return self._file_path

  def append_to(self, document: Report):
    self.save()
    close_figure(self.figure)
    document.append(StandAloneGraphic(filename=self.file_path))
    document.append(LineBreak())


  def save(self):
    self.figure.savefig(self.file_path)


# Package imports
from . import selector
from .anatomical_plot import AnatomicalPlot # Depends on Plot
from .triple_plot import TriplePlot # Depends on Plot And AnatomicalPlot
