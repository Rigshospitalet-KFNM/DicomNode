"""This module is build for visualizing a dicom image such that you can look
through it and see how the image looks"""

# Python standard library
from enum import IntEnum
from typing import Optional

# Third party Libraries
import numpy
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# Dicomnode modules
from dicomnode.math.image import Image, FramedImage

class Orientation(IntEnum):
    X = 0
    Y = 1
    Z = 2

class IndexTracker:
  @property
  def index(self):
    if self.orientation == Orientation.X:
      return (slice(None), slice(None), self._index)
    if self.orientation == Orientation.Y:
      return (slice(None), self._index, slice(None))
    if self.orientation == Orientation.Z:
      return (self._index, slice(None), slice(None))

  @property
  def max_frame(self):
    if self.X.ndim == 4:
      return self.X.shape[0] - 1
    else:
      return 0

  @property
  def max_index(self):
    match self.orientation:
      case Orientation.X:
        orientation_index = self.X.ndim - 1
      case Orientation.Y:
        orientation_index = self.X.ndim - 2
      case Orientation.Z:
        orientation_index = self.X.ndim - 3

    return self.X.shape[orientation_index] - 1

  @property
  def display_data(self) -> numpy.ndarray:
    if self.X.ndim == 4:
      index = (self.frame,) + self.index
      return self.X[index]
    else:
      return self.X[self.index]

  @property
  def mask_data(self):
    if self.mask is not None:
      return self.mask[self.index]

    return None

  def __init__(self, ax: Axes, X: numpy.ndarray, mask: Optional[numpy.ndarray]=None, orientation=Orientation.X):
    self.frame = 0
    self.mask = mask
    self.orientation = orientation
    self.X = X
    self._index = self.max_index // 2
    self.ax = ax
    self.ax.set_xticks([],[])
    self.ax.set_yticks([],[])
    self.im = ax.imshow(self.display_data, cmap='gray_r')
    self.update()

  def on_key(self, event):
    increment = 1 if event.key == 'a' else (-1 if event.key == 'd' else 0)
    self.frame = numpy.clip(self.frame + increment, 0, self.max_frame)
    self.update()

  def on_scroll(self, event):
    increment = 1 if event.button == 'up' else -1
    self._index = numpy.clip(self._index + increment, 0, self.max_index)
    self.update()

  def update(self):
    self.im.set_data(self.display_data)
    if self.mask is not None:
      self.ax.contour(self.mask_data)

    if self.max_frame != 0:
      self.ax.set_title(
          f'frame: {self.frame}\nindex {self._index}')
    else:
      self.ax.set_title(f'Index: {self._index}')
    self.im.axes.figure.canvas.draw()

def track_image_to_axes(
  figure: Figure,
  axes: Axes,
  image: Image | FramedImage,
  orientation = Orientation.Z) -> IndexTracker:

  tracker = IndexTracker(
    axes,
    image.raw,
    orientation=orientation
  )

  def on_key(event):
    tracker.on_key(event)

  figure.canvas.mpl_connect('key_press_event', on_key)

  def on_scroll(event):
    tracker.on_scroll(event)

  figure.canvas.mpl_connect('scroll_event', on_scroll)

  return tracker
