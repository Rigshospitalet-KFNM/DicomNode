from dataclasses import dataclass, field

from os import environ

from pathlib import Path
from pydicom import Dataset
from pylatex import Document, MiniPage, NoEscape, Package, Command, Head, Foot, \
  PageStyle, StandAloneGraphic
from pylatex.utils import bold
from typing import List, Tuple, Optional, Union

import logging

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
class StudyHeader:
  patient_name: str
  CPR: str
  study: str
  series: str
  date: str # Note this is not a date that is intended for calculation, but display

  @classmethod
  def from_dicom(cls, dicom: Dataset) -> 'StudyHeader':
    return cls(
      patient_name=dicom.PatientName,
      CPR=dicom.PatientID,
      study=dicom.StudyDescription,
      series=dicom.SeriesDescription,
      date=dicom.StudyDate.strftime("%d/%m/%Y")
    )

@dataclass
class HeaderData:
  icon_path: str
  hospital_name: str
  department: str
  address: str


@dataclass
class Conclusion:
  conclusion: Optional[str] = None
  key_numbers: List[Tuple[str, Union[float, int]]] = field(default_factory=list)

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

  def add_header(self, header_data: HeaderData):
    header = PageStyle("header")

    with header.create(Head('L')):
      icon_path = header_data.icon_path
      # Check if file exists
      header.append(StandAloneGraphic(filename=icon_path,
        image_options=NoEscape("width=120px")
      ))

    with header.create(Foot('L')):
      header.append(f"{header_data.hospital_name}\n")
      header.append(f"{header_data.department}\n")
      header.append(f"{header_data.address}\n")

    self.preamble.append(header)
    self.change_document_style("header")

  def add_study(self, study_header: StudyHeader):
    with self.create(MiniPage(width=NoEscape(r"0.47\textwidth"))):
      self.append(f"Navn: {bold(study_header.patient_name)}\n")
      self.append(f"CPR: {bold(study_header.CPR)}\n")
      self.append(f"Studie: {bold(study_header.study)}\n")
      self.append(f"Serie: {bold(study_header.series)}\n")
      self.append(f"Dato: {bold(study_header.date)}\n")

  def add_conclusion(self, conclusion: Conclusion):
    if conclusion is not None:
      with self.create(MiniPage(width=NoEscape(r"0.47\textwidth"))):
        if conclusion.conclusion is not None:
          self.append(f"{conclusion.conclusion} \n")

        for name, value in conclusion.key_numbers:
          self.append(f"{name}: {bold(str(value))}")

