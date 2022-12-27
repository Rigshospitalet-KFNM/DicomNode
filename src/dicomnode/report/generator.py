from dataclasses import dataclass, field
from enum import Enum
from os import environ

from pathlib import Path
from pydicom import Dataset
from pylatex import Document, MiniPage, NoEscape, Package, Command, Head, Foot, \
  PageStyle, StandAloneGraphic, Tabular as LatexTable
from pylatex.utils import bold
from typing import List, Tuple, Optional, Union

import logging

import platform

from shutil import which

from dicomnode.constants import DICOMNODE_ENV_FONT_PATH
from dicomnode.lib.exceptions import InvalidLatexCompiler, InvalidFont

compilers: List[str] = [
  "pdflatex",
  "lualatex",
  "xelatex",
  "latex",
]

logger = logging.getLogger("dicomnode")

@dataclass
class PatientHeader:
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

@dataclass
class DocumentHeader:
  icon_path: str
  hospital_name: str
  department: str
  address: str


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

  def __init__(
      self,
      file_name: str,
      options = Options()
    ) -> None:
    super().__init__(file_name, geometry_options={
        'tmargin' : self.options.margin_top,
        'lmargin' : self.options.margin_side
    })

    self.options = options
    self.file_name = file_name

    if self.options.compiler == "default":
      if DICOMNODE_ENV_FONT_PATH in environ:
        self.load_font(environ[DICOMNODE_ENV_FONT_PATH])
      elif self.options.font is not None:
        self.load_font(self.options.font)
      else:
        self.compiler = "pdflatex"
    else:
      self.compiler = self.options.compiler

    if self.compiler not in compilers:
      raise InvalidLatexCompiler

    if which(self.compiler) is None:
      logger.error(f"{self.compiler} is not found.")
      if platform.system() == 'Linux':
        logger.error("Dependant on your Linux distribution you can install the needed compiler with:")
        logger.error("sudo apt install texlive-full")
        logger.error("sudo yum install texlive-*")

      raise InvalidLatexCompiler


  def load_font(self, font_path_str: str) -> None:
    """Loads a font into the document

    Args:
        font_path_str (str): String with path to font .oft or .ttf

    Raises:
        InvalidFont: When font is not a oft or tff font
        FileNotFoundError: when font is not found
    """
    if not (font_path_str.lower().endswith('.oft') or font_path_str.lower().endswith('.ttf')):
      raise InvalidFont

    font_path = Path(font_path_str)

    if not font_path.exists():
      logger.error("Font file not found")
      raise FileNotFoundError()

    self.compiler = "xelatex"
    self.packages.append(Package("fontspec"))
    self.preamble.append(Command("setmainfont",NoEscape(rf"{font_path_str}")))

  def add_document_header(self, document_header: DocumentHeader):
    """Adds a standardized document header to a document

    Args:
        document_header (DocumentHeader): header to be added
    """
    header = PageStyle("header")

    with header.create(Head('L')):
      icon_path = document_header.icon_path
      # Check if file exists
      header.append(StandAloneGraphic(filename=icon_path,
        image_options=NoEscape("width=120px")
      ))

    with header.create(Foot('L')):
      header.append(f"{document_header.hospital_name}\n")
      header.append(f"{document_header.department}\n")
      header.append(f"{document_header.address}\n")

    self.preamble.append(header)
    self.change_document_style("header")

  def add_danish_patient_header(self, patient_header: PatientHeader):
    """Adds a mini page with basic patient information in the danish language

    Args:
        patient_header (PatientHeader): patient header to be added
    """
    with self.create(MiniPage(width=NoEscape(r"0.47\textwidth"))):
      self.append(f"Navn: {bold(patient_header.patient_name)}\n")
      self.append(f"CPR: {bold(patient_header.CPR)}\n")
      self.append(f"Studie: {bold(patient_header.study)}\n")
      self.append(f"Serie: {bold(patient_header.series)}\n")
      self.append(f"Dato: {bold(patient_header.date)}\n")

  def add_patient_header(self, patient_header: PatientHeader):
    """Adds a mini page with basic patient information

    Args:
        patient_header (PatientHeader): patient header to be added
    """
    with self.create(MiniPage(width=NoEscape(r"0.47\textwidth"))):
      self.append(f"Name: {bold(patient_header.patient_name)}\n")
      self.append(f"ID: {bold(patient_header.CPR)}\n")
      self.append(f"Study: {bold(patient_header.study)}\n")
      self.append(f"Series: {bold(patient_header.series)}\n")
      self.append(f"Date: {bold(patient_header.date)}\n")

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

