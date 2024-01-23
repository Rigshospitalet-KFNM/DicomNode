"""This module provides a standardized way of producing figures, plots and
various displays for a standardized way to present the reports"""



# Python3 Standard Library
from abc import ABC, abstractmethod
from enum import Enum
import os
from typing import Any, Sequence

# Third party packages
from pylatex import StandAloneGraphic, MiniPage, NoEscape
import numpy
import nibabel
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LinearLocator

# Dicomnode packages
from dicomnode.report.generator import LaTeXComponent, Report

class AxisTypes(Enum):
  NO_AXIS = 0
  ISO_CENTER = 1
  PURE_DISTANCE = 2


class AnatomicalPlane(Enum):
  SAGITTAL=0
  CORONAL=1
  TRANSVERSE=2

class PlaneImages(Sequence):
  def __init__(self, image: numpy.ndarray, plane: AnatomicalPlane) -> None:
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
      if self.plane == AnatomicalPlane.SAGITTAL:
        return self.image[start:stop:step, :, :]
      if self.plane == AnatomicalPlane.CORONAL:
        return self.image[:, start:stop:step, :]
      if self.plane == AnatomicalPlane.TRANSVERSE:
        return self.image[:, :, start:stop:step]

    if self.plane == AnatomicalPlane.SAGITTAL:
      return self.image[index, :, :]
    if self.plane == AnatomicalPlane.CORONAL:
      return self.image[:, index, :]
    if self.plane == AnatomicalPlane.TRANSVERSE:
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


class Selector(ABC):
  def __init__(self):
    pass

  @abstractmethod
  def __call__(self, images: Sequence[numpy.ndarray]) -> Any:
    pass

class PercentageSelector(Selector):
  def __init__(self, percentage = 50):
    super().__init__()
    if isinstance(percentage, int):
      self.percentage = percentage / 100
    else:
      self.percentage = percentage

  def __call__(self, images: Sequence[numpy.ndarray]):
    middle = int(len(images) * self.percentage)
    return images[middle], middle


class Plot(LaTeXComponent):
  def __init__(self, file_path) -> None:
    self._file_path = file_path
    self.figure = plt.figure()

  def append_to(self, document: Report):
    self.save()

    with document.create(MiniPage(width=NoEscape(r"\textwidth"))) as mini_page:
      mini_page.append(StandAloneGraphic(filename=self._file_path))



  def save(self):
    self.figure.savefig(self._file_path)


class TriplePlot(Plot):
  """This plot consists of plot of each plane

  """

  def __init__(self, 
               file_path, 
               nifti_image: nibabel.nifti1.Nifti1Image,
               selector: Selector = PercentageSelector()) -> None:
    super().__init__(file_path)
    self.figure.set_figheight(6.0)
    self.figure.set_figwidth(15.0)

    self.nifti_image = nifti_image # this should get removed, and only store what we need
    self.selector = selector

    self.sagittal_axes = self.figure.add_subplot(1,3,1)
    self.coronal_axes = self.figure.add_subplot(1,3,2)
    self.transverse_axes = self.figure.add_subplot(1,3,3)

    image_data = nifti_image.get_fdata()
    print(image_data.shape)

    sagittal_image, sagittal_index = self.selector(PlaneImages(image_data, AnatomicalPlane.SAGITTAL))
    coronal_image, coronal_index = self.selector(PlaneImages(image_data, AnatomicalPlane.CORONAL))
    transverse_image, transverse_index = self.selector(PlaneImages(image_data, AnatomicalPlane.TRANSVERSE))

    if self.nifti_image.affine is not None:
      transverse_vector = self.nifti_image.affine[0, :3]
      coronal_vector =  self.nifti_image.affine[1, :3]
      sagittal_vector =  self.nifti_image.affine[2, :3]

      def sagittal_format(value, tick_number):
        return f"{numpy.floor(numpy.linalg.norm(sagittal_vector * value)):g} mm"

      def coronal_format(value, tick_number):
        return f"{numpy.floor(numpy.linalg.norm(coronal_vector * value)):g} mm"

      def transverse_format(value, tick_number):
        return f"{numpy.floor(numpy.linalg.norm(transverse_vector * value)):g} mm"

      self.sagittal_axes.yaxis.set_major_locator(LinearLocator(5))
      self.sagittal_axes.xaxis.set_major_locator(LinearLocator(5))

      self.coronal_axes.yaxis.set_major_locator(LinearLocator(5))
      self.coronal_axes.xaxis.set_major_locator(LinearLocator(5))

      self.transverse_axes.yaxis.set_major_locator(LinearLocator(5))
      self.transverse_axes.xaxis.set_major_locator(LinearLocator(5))

      self.sagittal_axes.xaxis.set_tick_params(rotation=90)
      self.coronal_axes.xaxis.set_tick_params(rotation=90)
      self.transverse_axes.xaxis.set_tick_params(rotation=90)

      self.sagittal_axes.xaxis.set_major_formatter(FuncFormatter(transverse_format))
      self.sagittal_axes.yaxis.set_major_formatter(FuncFormatter(coronal_format))

      self.coronal_axes.xaxis.set_major_formatter(FuncFormatter(transverse_format))
      self.coronal_axes.yaxis.set_major_formatter(FuncFormatter(sagittal_format))

      self.transverse_axes.xaxis.set_major_formatter(FuncFormatter(coronal_format))
      self.transverse_axes.yaxis.set_major_formatter(FuncFormatter(sagittal_format))

    self.sagittal_axes.imshow(sagittal_image.T, cmap="grey", origin="lower")
    self.coronal_axes.imshow(coronal_image.T, cmap="grey", origin="lower")
    self.transverse_axes.imshow(transverse_image.T, cmap="grey", origin="lower")







