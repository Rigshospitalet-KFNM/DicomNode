"""Test cases for generating latex report

In general the test case will be checked on the generated LaTeX code.
Test results can be found in /tmp/dicomnode
"""

# Python3 Standard Library imports
from datetime import datetime as DateTime
from pathlib import Path
from unittest import TestCase

# Third party imports
import nibabel
from pydicom import Dataset

# Dicomnode Imports
from dicomnode import library_paths
from dicomnode.report import Report
from dicomnode.report.latex_components import PatientInformation, ReportHeader

from dicomnode.report.plot.triple_plot import TriplePlot

nifti_image: nibabel.nifti1.Nifti1Image = nibabel.loadsave.load(f'{library_paths.report_data_directory}/someones_anatomy.nii.gz') # type: ignore

class GeneratorTestCase(TestCase):
  def test_empty_report(self):
    test_file = f"{library_paths.report_directory}/test_empty_file"
    report = Report(test_file)
    report.generate_tex() # Appends ".tex"

    with open(f"{test_file}.tex",'r') as fp:
      raw_tex_content = fp.read()

    # Assert once there's a stable interface

  def test_report_header(self):
    dataset = Dataset()

    dataset.PatientName = r"Familiy Name^Test name"
    dataset.PatientID = "XXXXXX-XXXX"
    dataset.StudyDescription = "Study Test"
    dataset.SeriesDescription = "Series Test"
    dataset.StudyDate = DateTime(2020,1,23)

    patient_header = PatientInformation.from_dicom(dataset)


    document_header = ReportHeader(
      icon_path=f"{library_paths.report_data_directory}/report_image.png",
      lines=["test_hospital", "test department", "Test address"]
    )

    test_header_doc = f"{library_paths.report_directory}/test_doc"
    triple_plot_options = TriplePlot.Options(file_path=f"{library_paths.figure_directory}/report_figure.png")

    triple_plot = TriplePlot(nifti_image, triple_plot_options)

    report = Report(test_header_doc)
    report.append(document_header)
    report.append(patient_header)
    report.append(triple_plot)

    report.generate_tex()

    with open(f"{test_header_doc}.tex",'r') as fp:
      raw_tex_content = fp.read()
      print(raw_tex_content)

    report.generate_pdf()


