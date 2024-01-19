"""Test cases for generating latex report

In general the test case will be checked on the generated LaTeX code.
"""

# Python3 Standard Library imports
from datetime import datetime as DateTime
from pathlib import Path
from unittest import TestCase

# Third party imports
from pydicom import Dataset

# Dicomnode Imports
from dicomnode import library_paths
from dicomnode.report import generator

class GeneratorTestCase(TestCase):
  def test_empty_report(self):
    test_file = f"{library_paths.report_directory}/test_empty_file"
    report = generator.Report(test_file)
    report.generate_tex() # Appends ".tex"

    with open(f"{test_file}.tex",'r') as fp:
      raw_tex_content = fp.read()

    # Assert once there's a stable interface

  def test_report_header(self):
    dataset = Dataset()

    dataset.PatientName = "Familiy Name^Test name"
    dataset.PatientID = "XXXXXX-XXXX"
    dataset.StudyDescription = "Study Test"
    dataset.SeriesDescription = "Series Test"
    dataset.StudyDate = DateTime(2020,1,23)

    patient_header = generator.PatientHeader.from_dicom(dataset)


    document_header = generator.DocumentHeader(
      icon_path=f"{library_paths.report_data_directory}/report_image.png",
      hospital_name="test_hospital",
      department="test department",
      address="Test address"
    )

    test_header_doc = f"{library_paths.report_directory}/test_doc"

    report = generator.Report(test_header_doc)
    report.add_document_header(document_header) # This is a problematic design
    report.add_danish_patient_header(patient_header)
    report.generate_tex()

    with open(f"{test_header_doc}.tex",'r') as fp:
      raw_tex_content = fp.read()
      print(raw_tex_content)

    report.generate_pdf()

