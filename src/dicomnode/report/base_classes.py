"""This library contains various base classes used by this library
It is unlikely you need anything from here unless you are extending
base classes
  """

# Python Base Classes
from abc import ABC, abstractmethod
from enum import Enum
from typing import Sequence

# Third Party Packages
from numpy import ndarray

# Dicomnode packages
from dicomnode import report


# List of known Latex compilers
class LaTeXCompilers(Enum):
  DEFAULT = "default"
  PDFLATEX = "pdflatex"
  LUALATEX = "lualatex"
  XELATEX = "xelatex"
  LATEX = "latex"

class LaTeXComponent(ABC):
  @abstractmethod
  def append_to(self, document: 'report.Report'):
    raise NotImplemented #pragma: ignore

  @classmethod
  def from_dicom(cls):
    raise NotImplemented # pragma: no cover

class Selector(ABC):
  def __init__(self):
    pass

  @abstractmethod
  def __call__(self, images: Sequence[ndarray]):
    pass

