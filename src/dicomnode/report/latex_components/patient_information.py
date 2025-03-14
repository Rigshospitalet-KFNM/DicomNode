"""Module for the latex component patient header"""

# Python Standard Library
from dataclasses import dataclass

# Third party Packages
from pydicom import Dataset
from pydicom.valuerep import PersonName
from pylatex import MiniPage, NoEscape, Package, MdFramed, HFill
from pylatex.utils import bold, escape_latex

# Dicomnode Packages
from dicomnode.dicom import format_from_patient_name
from dicomnode.report import Report, add_line
from dicomnode.report.base_classes import LaTeXComponent
from dicomnode.report.latex_components.dicom_frame import DicomFrame

@dataclass
class PatientInformation(LaTeXComponent):
  patient_name: PersonName
  CPR: str
  study: str
  series: str
  date: str # Note this is not a date that is intended for display, not calculation

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
    with report.create(DicomFrame()) as frame:
      add_line(frame, "Navn: ", HFill(), bold(escape_latex(format_from_patient_name(self.patient_name))))
      add_line(frame, "CPR: ", HFill(), bold(escape_latex(self.CPR)))
      add_line(frame, "Studie: ", HFill(), bold(escape_latex(self.study)))
      add_line(frame, "Serie: ", HFill(), bold(escape_latex(self.series)))
      frame.append('Dato: ')
      frame.append(HFill())
      frame.append(bold(escape_latex(self.date)))
