""""""

__author__ = "Christoffer Vilstrup Jensen"

# Python standard Library
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from os import environ
from pathlib import Path
from typing import List, Tuple, Optional, Union
import platform
from shutil import which

# Third party Packages
from pydicom import Dataset
from pylatex.base_classes import Container
from pylatex import Document, MiniPage, NoEscape, Package, Command, Head, Foot, \
  PageStyle, StandAloneGraphic, Tabular as LatexTable, LineBreak
from pylatex.utils import bold

# Dicomnode packages
from dicomnode.constants import DICOMNODE_ENV_FONT
from dicomnode.lib.exceptions import InvalidLatexCompiler, InvalidFont
from dicomnode.lib.logging import get_logger

# List of known Latex compilers
compilers: List[str] = [
  "pdflatex",
  "lualatex",
  "xelatex",
  "latex",
]

logger = get_logger()

def add_line(container: Container, *args):
  for arg in args:
    container.append(arg)
  container.append(LineBreak())

@dataclass
class LaTeXComponent(ABC):
  @abstractmethod
  def append_to(self, document: 'Report'):
    raise NotImplemented #pragma: ignore

@dataclass
class PatientHeader(LaTeXComponent):
  patient_name: str
  CPR: str
  study: str
  series: str
  date: str # Note this is not a date that is intended for calculation, but display

  @classmethod
  def from_dicom(cls, dicom: Dataset) -> 'PatientHeader':
    return cls(
      patient_name=dicom.PatientName,
      CPR=dicom.PatientID,
      study=dicom.StudyDescription,
      series=dicom.SeriesDescription,
      date=dicom.StudyDate.strftime("%d/%m/%Y")
    )

  def append_to(self, document: 'Report'):
    """Adds a mini page with basic patient information in the danish language

    Args:
        patient_header (PatientHeader): patient header to be added
    """
    with document.create(MiniPage(width=NoEscape(r"0.49\textwidth"), align='l')) as mini_page:
      add_line(mini_page, "Navn: ", bold(self.patient_name))
      add_line(mini_page, "CPR: ", bold(self.CPR))
      add_line(mini_page, "Studie: ", bold(self.study))
      add_line(mini_page, "Serie: ", bold(self.series))
      add_line(mini_page, "Dato: ", bold(self.date))

@dataclass
class DocumentHeader(LaTeXComponent):
  icon_path: str
  hospital_name: str
  department: str
  address: str

  def append_to(self, document: 'Report'):
    """Adds a standardized document header to a document

    Args:
        document (Document): Report that this document header is added to.
    """
    header = PageStyle("header", header_thickness='0.5')

    with header.create(Head('L')) as header_left:
      with header_left.create(MiniPage(width=NoEscape(r"0.49\textwidth"))) as wrapper:
        icon_path = self.icon_path
        wrapper.append(StandAloneGraphic(filename=icon_path,
          image_options=NoEscape("width=120pt")
        ))

    with header.create(Head('R')) as header_right:
      with header_right.create(MiniPage(width=NoEscape(r"0.49\textwidth"), pos='r', align='r')) as wrapper:
        add_line(wrapper, self.hospital_name)
        add_line(wrapper, self.department)
        add_line(wrapper, self.address)

    document.preamble.append(header)
    document.change_document_style("header")


@dataclass
class Conclusion:
  conclusion: Optional[str] = None
  key_numbers: List[Tuple[str, Union[float, int]]] = field(default_factory=list)

class TableStyle(Enum):
  FULL = 1
  BORDER = 2
  TOP_BOTTOM = 3
  none = 4

@dataclass
class Table:
  table_style: TableStyle
  withHeader: bool = True
  Alignment: List[str] = field(default_factory=list)
  Rows: List[List[str]] = field(default_factory=list)


class Report(Document):
  @dataclass
  class Options:
    margin_top: str = "2cm"
    margin_side: str = "2cm"
    compiler: str = "default"
    font: Optional[str] = None

  def append(self, other):
    if isinstance(other, LaTeXComponent):
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

    self.file_name = file_name
    self.__options = options
    if options.compiler == "default":
      if DICOMNODE_ENV_FONT in environ:
        self.load_font(environ[DICOMNODE_ENV_FONT])
      elif self.__options.font is not None:
        self.load_font(self.__options.font)
      else:
        self.compiler = "pdflatex"
    else:
      self.compiler = self.__options.compiler

    if self.compiler not in compilers:
      raise InvalidLatexCompiler

    if which(self.compiler) is None:
      logger.error(f"{self.compiler} is not found.")
      if platform.system() == 'Linux':
        logger.error("Dependant on your Linux distribution you can install the needed compiler with:")
        logger.error("sudo apt install texlive-full")
        logger.error("sudo yum install texlive-*")

      raise InvalidLatexCompiler


  def load_font(self, font: str) -> None:
    """Loads a font into the document

    Args:
        font_path_str (str): String with path to font .oft or .ttf

    Raises:
        InvalidFont: When font is not a oft or tff font
        FileNotFoundError: when font is not found
    """

    self.compiler = "xelatex"
    self.packages.append(Package("fontspec"))
    self.preamble.append(Command("setmainfont", NoEscape(rf"{font}"), options=[]))


  def add_conclusion(self, conclusion: Conclusion):
    """_summary_

    Args:
        conclusion (Conclusion): _description_
    """
    with self.create(MiniPage(width=NoEscape(r"0.47\textwidth"))):
      if conclusion.conclusion is not None:
        self.append(f"{conclusion.conclusion} \n")

      for name, value in conclusion.key_numbers:
        self.append(f"{name}: {bold(str(value))}")

  def add_table(self, table: Table):
    """_summary_

    Args:
        table (Table): _description_
    """
    if table.table_style == TableStyle.FULL:
      alignment = "| " + " | ".join(table.Alignment) + " |"
    if table.table_style == TableStyle.BORDER:
      alignment = "| " + " ".join(table.Alignment) + " |"
    else:
      alignment = " ".join(table.Alignment)

    with self.create(LatexTable(alignment)) as tab:
      tab: LatexTable = tab # Just there to make intellisense happy
      if table.table_style in [TableStyle.FULL, TableStyle.BORDER, TableStyle.TOP_BOTTOM]:
        tab.add_hline()

      if table.withHeader:
        headerRow = list(map(bold, table.Rows[0]))
      else:
        headerRow = table.Rows[0]

      tab.add_row(headerRow)

      if table.table_style in [TableStyle.FULL, TableStyle.BORDER, TableStyle.TOP_BOTTOM]:
        tab.add_hline()

      for row in table.Rows[1:]:
        tab.add_row(row)

        if table.table_style == TableStyle.FULL:
          tab.add_hline()

      # Full is missing due to line already being added
      if table.table_style in [TableStyle.BORDER, TableStyle.TOP_BOTTOM]:
        tab.add_hline()

    def add_plot(self):
      raise NotImplemented