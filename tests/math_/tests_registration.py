# Python Standard Library
from pathlib import Path
from unittest import skipIf

# Third party modules
import nibabel
import matplotlib.pyplot as plt

# Dicomnode modules
from dicomnode import library_paths
from dicomnode.dicom.series import extract_image
from dicomnode.math.image import Image, mask_image
from dicomnode.math import CUDA, bounding_box
from dicomnode.report.plot.triple_plot import TriplePlot
from dicomnode.report.plot.selector import AverageSelector

# Test
from tests.helpers.dicomnode_test_case import DicomnodeTestCase

ct_image_path = library_paths.report_data_directory / "CT_nifti" / "CT.nii"
ct_brain_path = library_paths.report_data_directory / "CT_nifti" / "segmentation" / "brain.nii.gz"
mniBrain_path = library_paths.report_data_directory / "tpl-MNI152Lin" / "tpl-MNI152Lin_02_PD.nii.gz"

files_exists = ct_brain_path.exists() and ct_brain_path.exists() and mniBrain_path.exists()

class RegistrationTestCase(DicomnodeTestCase):
  @skipIf((not CUDA) or (not files_exists), "Need GPU and files")
  def test_registration_from_python(self):
    from dicomnode.math import _cuda

    nifti: nibabel.nifti1.Nifti1Image = nibabel.loadsave.load(ct_image_path) # type: ignore
    image = extract_image(nifti)

    seg_nifti: nibabel.nifti1.Nifti1Image = nibabel.loadsave.load(ct_brain_path) # type: ignore

    seg_image = extract_image(seg_nifti)

    mni: nibabel.nifti1.Nifti1Image = nibabel.loadsave.load(mniBrain_path) # type: ignore

    hmmm = mask_image(image, seg_image)

    print(_cuda.center_of_gravity(hmmm))


    figure = TriplePlot(hmmm.raw, figure=plt.figure(), options=TriplePlot.Options(
      selector=AverageSelector()
    ))

    plt.show()
