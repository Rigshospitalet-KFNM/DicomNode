# Python Standard library


# Third party packages
from nibabel import nifti1

# Dicomnode Modules
from dicomnode.dicom import series
from dicomnode.dicom.nifti import convert_to_nifti
from tests.helpers.dicomnode_test_case import DicomnodeTestCase
from tests.helpers import generate_numpy_datasets

class DicomnodeDicomNifti(DicomnodeTestCase):
  def test_can_nifti(self):
    datasets = series.DicomSeries([ds for ds in generate_numpy_datasets(10, Rows=10, Cols=10)])

    datasets["Modality"] = "CT"

    nifti = convert_to_nifti(datasets.datasets, None, False)
    self.assertIsInstance(nifti, nifti1.Nifti1Image)
