"""Test cases for generating latex report

In general the test case will be checked on the generated LaTeX code.
Test results can be found in /tmp/dicomnode
"""

# Python3 Standard Library imports
from datetime import datetime as DateTime
from pathlib import Path
from unittest import TestCase, skipIf

# Third party imports
import nibabel
from pydicom import Dataset
from pydicom.uid import SecondaryCaptureImageStorage

# Dicomnode Imports
from dicomnode import library_paths
from dicomnode.lib.io import save_dicom
from dicomnode.dicom import generate_uid, make_meta
from dicomnode.dicom.blueprints import default_report_blueprint
from dicomnode.dicom.dicom_factory import DicomFactory
from dicomnode.report import Report
from dicomnode.report.latex_components.dicom_frame import DicomFrame
from dicomnode.report.latex_components import PatientInformation, ReportHeader, Table

from dicomnode.report.plot.triple_plot import TriplePlot

nifti_path = Path(f'{library_paths.report_data_directory}/someones_anatomy.nii.gz')
figure_image_path = Path(f"{library_paths.report_data_directory}/report_image.png")

if nifti_path.exists():
  nifti_image: nibabel.nifti1.Nifti1Image = nibabel.loadsave.load(f'{library_paths.report_data_directory}/someones_anatomy.nii.gz') # type: ignore
else:
  nifti_image = None

class GeneratorTestCase(TestCase):
  def test_empty_report(self):
    test_file = f"{library_paths.report_directory}/test_empty_file"
    report = Report(test_file)
    report.generate_tex() # Appends ".tex"

    with open(f"{test_file}.tex",'r') as fp:
      raw_tex_content = fp.read()

    # Assert once there's a stable interface

  @skipIf(not nifti_path.exists() or not figure_image_path.exists(), "Needs an image to plot")
  def test_report(self):
    dataset = Dataset()

    dataset.PatientName = r"Familiy Name^Test name"
    dataset.SOPInstanceUID = generate_uid()
    dataset.SOPClassUID = SecondaryCaptureImageStorage
    dataset.PatientID = "XXXXXX-XXXX"
    dataset.StudyDescription = "Study Test"
    dataset.SeriesDescription = "Series Test"
    dataset.StudyDate = DateTime(2020,1,23)
    dataset.InstitutionalDepartmentName = "Department"
    dataset.InstitutionName = "Hospital"
    dataset.InstitutionAddress = "Department address"
    dataset.AccessionNumber = "TEST_ACCESSION"
    dataset.StudyInstanceUID = generate_uid()

    patient_header = PatientInformation.from_dicom(dataset)


    document_header = ReportHeader.from_dicom(
      icon_path=f"{library_paths.report_data_directory}/report_image.png",
      dataset=dataset
    )

    test_header_doc = f"{library_paths.report_directory}/test_doc"
    triple_plot_options = TriplePlot.Options(file_path=f"{library_paths.figure_directory}/report_figure.png")
    triple_plot = TriplePlot(nifti_image, triple_plot_options)

    table = Table(Table.TableStyle.FULL, Alignment=['l', 'c', 'r', 'X', 'X'], Rows=[
      ["Hello", "World", "I should put a grafic in here", 'Bla?', "Bla!"],
      ["Hello", "World", "I should put a grafic in here", 'Bla?', "Bla!"],
    ])

    dicom_frame = DicomFrame()

    dicom_frame.append("Bla bla bla")
    dicom_frame.append("Bla bla bla")
    dicom_frame.append("Bla bla bla")


    report = Report(test_header_doc)
    report.append(document_header)
    report.append(patient_header)
    report.append(triple_plot)
    report.append(table)
    report.append(dicom_frame)
    report.generate_tex()

    with open(report.file_name + '.tex') as fp:
      #print(fp.read())
      pass

    report.generate_pdf()
    factory = DicomFactory()

    encoded_report = factory.encode_pdf(report, [dataset], default_report_blueprint)
    make_meta(encoded_report)

    save_dicom(library_paths.report_directory/'test_report.dcm', encoded_report)


