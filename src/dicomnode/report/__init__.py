""""""

# Python Standard packages
from dataclasses import dataclass as _dataclass
from typing import List as _List, Optional as _Optional, Type as _Type
from os import environ as _environ
import platform as _platform
from shutil import which as _which

# Third party packages
from pydicom import Dataset
from pylatex import Command as _Command, Document as _Document, LineBreak as _LineBreak, NoEscape as _NoEscape, Package as _Package
from pylatex.base_classes import Container as _Container

# Private dicomnode imports
from dicomnode.constants import DICOMNODE_ENV_FONT as _DICOMNODE_ENV_FONT
from dicomnode.lib.logging import get_logger as _get_logger
from dicomnode.lib.exceptions import InvalidLatexCompiler as _InvalidLatexCompiler

from . import base_classes


def add_line(container: _Container, *args):
  for arg in args:
    container.append(arg)
  container.append(_LineBreak())

class Report(_Document):
  """
  Latex report for a study, the report is blank initially

  Classes:
    Options: Dataclass for various options

  Methods:
    append: Method for adding additional content to the report.
    generate_pdf: Generate the pdf file

  Attributes:
    packages (List[Pylatex.Package]): LaTeX packages added to the LaTeX document
  """
  @_dataclass
  class Options:
    margin_top: str = "2cm"
    margin_side: str = "2cm"
    compiler: base_classes.LaTeXCompilers = base_classes.LaTeXCompilers.DEFAULT
    font: _Optional[str] = None

  def append(self, other):
    if isinstance(other, base_classes.LaTeXComponent):
      other.append_to(self)
    else:
      super().append(other)

  def __init__(
      self,
      file_name: str,
      options = Options()
    ) -> None:
    #self.options = options
    super().__init__(file_name, geometry_options={
        'tmargin' : options.margin_top,
        'lmargin' : options.margin_side,
        "includeheadfoot" : True,
        "head": "40pt"
    })
    logger = _get_logger()

    self.file_name = file_name
    self.__options = options
    if options.compiler == base_classes.LaTeXCompilers.DEFAULT:
      if _DICOMNODE_ENV_FONT in _environ:
        # load_font sets self.compiler
        self.__load_font(_environ[_DICOMNODE_ENV_FONT]) # pragma: no cover
      elif self.__options.font is not None:
        self.__load_font(self.__options.font) # pragma: no cover
      else:
        self.__compiler = base_classes.LaTeXCompilers.PDFLATEX # pragma: no cover
    else:
      self.__compiler = self.__options.compiler # pragma: no cover

    if _which(self.__compiler.value) is None: # pragma: no cover
      logger.error(f"{self.__compiler} is not found in PATH, Either install it or update your PATH environment variable")
      if _platform.system() == 'Linux':
        logger.error("Dependant on your Linux distribution you can install the needed compiler with:")
        logger.error("sudo apt install texlive-full")
        logger.error("sudo yum install texlive-*")
      raise _InvalidLatexCompiler

  def generate_pdf(self, filepath=None, *, clean=True, clean_tex=True, compiler_args=None, silent=True):
    logger = _get_logger()
    logger.info(f"Calling with compiler: {self.__compiler.value}")
    # Note that the super class will read from class
    return super().generate_pdf(filepath,
                                clean=clean,
                                clean_tex=clean_tex,
                                compiler=self.__compiler.value,
                                compiler_args=compiler_args,
                                silent=silent)

  def __load_font(self, font: str) -> None:
    """Loads a font into the document

    Args:
        font_path_str (str): String with path to font .oft or .ttf

    Raises:
        InvalidFont: When font is not a oft or tff font
        FileNotFoundError: when font is not found
    """
    # Note this function is not covered as it's dependant on environment
    self.__compiler = base_classes.LaTeXCompilers.XELATEX # pragma: no cover
    self.packages.append(_Package("fontspec", options=[_NoEscape("no-math")])) # pragma: no cover
    self.packages.append(_Package("mathspec")) # pragma: no cover
    self.preamble.append(_Command("setmainfont", _NoEscape(rf"{font}"), options=[])) # pragma: no cover
    self.preamble.append(_Command("setsansfont", _NoEscape(rf"{font}"), options=[])) # pragma: no cover
    self.preamble.append(_Command("setmonofont", _NoEscape(rf"{font}"), options=[])) # pragma: no cover
    self.preamble.append(_Command("setmathfont", _NoEscape(rf"{font}"), options=[])) # pragma: no cover

# Dicomnode packages
from . import plot
