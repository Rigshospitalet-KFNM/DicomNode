"""Module for the patient header

  """

# Python Standard Library
from dataclasses import dataclass

# Third party Packages
from pydicom import Dataset
from pylatex import MiniPage, NoEscape, Package
from pylatex.utils import bold

# Dicomnode Packages
from dicomnode.report import Report, add_line
from dicomnode.report.pylatex_extensions.framed import Framed
from dicomnode.report.base_classes import LaTeXComponent

@dataclass
class PatientInformation(LaTeXComponent):
  patient_name: str
  CPR: str
  study: str
  series: str
  date: str # Note this is not a date that is intended for calculation, but display

  @classmethod
  def from_dicom(cls, dicom: Dataset) -> 'PatientInformation':
    return cls(
      patient_name=dicom.PatientName,
      CPR=dicom.PatientID,
      study=dicom.StudyDescription,
      series=dicom.SeriesDescription,
      date=dicom.StudyDate.strftime("%d/%m/%Y")
    )

  def append_to(self, report: Report):
    """Adds a mini page with basic patient information in the danish language

    Args:
        patient_header (PatientHeader): patient header to be added
    """


    with report.create(Framed()) as frame:
      with frame.create(MiniPage(width=NoEscape(r"0.49\textwidth"), align='l')) as mini_page:
        add_line(mini_page, "Navn: ", bold(self.patient_name))
        add_line(mini_page, "CPR: ", bold(self.CPR))
        add_line(mini_page, "Studie: ", bold(self.study))
        add_line(mini_page, "Serie: ", bold(self.series))
        add_line(mini_page, "Dato: ", bold(self.date))