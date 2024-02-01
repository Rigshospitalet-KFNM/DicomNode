"""This library contains various base classes used by this library
It is unlikely you need anything from here unless you are extending
base classes
  """

# Python Base Classes
from abc import ABC, abstractmethod
from enum import Enum
from typing import Sequence, Iterable, Union

# Third Party Packages
from pydicom import Dataset
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
    """Add the content of this component to a report

    Args:
        document (report.Report): _description_

    Raises:
        NotImplemented: _description_
    """
    raise NotImplemented #pragma: ignore

  @classmethod
  def from_dicom(cls, dataset: Union[Iterable[Dataset],Dataset]):
    raise NotImplemented # pragma: no cover

class Selector(ABC):
  def __init__(self):
    pass

  @abstractmethod
  def __call__(self, images: Sequence[ndarray]):
    pass

