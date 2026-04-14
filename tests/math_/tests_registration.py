# Python Standard Library
from pathlib import Path
from unittest import skipIf

# Third party modules
import numpy
import nibabel
import matplotlib.pyplot as plt

# Dicomnode modules
from dicomnode import library_paths
from dicomnode.dicom.series import extract_image
from dicomnode.math import CUDA, bounding_box, center_of_gravity, cpu_center_of_gravity
from dicomnode.math.image import Image, mask_image
from dicomnode.report.plot.triple_plot import TriplePlot
from dicomnode.report.plot.selector import AverageSelector

# Test
from tests.helpers.dicomnode_test_case import DicomnodeTestCase

ct_image_path = library_paths.report_data_directory / "CT_nifti" / "CT.nii"
ct_brain_path = library_paths.report_data_directory / "CT_nifti" / "segmentation" / "brain.nii.gz"
mniBrain_path = library_paths.report_data_directory / "tpl-MNI152Lin" / "tpl-MNI152Lin_res-02_PD.nii.gz"

files_exists = ct_brain_path.exists() and ct_brain_path.exists() and mniBrain_path.exists()

class RegistrationTestCase(DicomnodeTestCase):
  @skipIf((not CUDA) or (not files_exists), "Need GPU and files")
  def test_registration_from_python(self):

    nifti: nibabel.nifti1.Nifti1Image = nibabel.loadsave.load(ct_image_path) # type: ignore
    image = extract_image(nifti)

    seg_nifti: nibabel.nifti1.Nifti1Image = nibabel.loadsave.load(ct_brain_path) # type: ignore

    seg_image = extract_image(seg_nifti)

    mni: nibabel.nifti1.Nifti1Image = nibabel.loadsave.load(mniBrain_path) # type: ignore
    mni_image = extract_image(mni)

    masked_image = mask_image(image, seg_image)

    mask_cog_index = center_of_gravity(masked_image)
    template_cog_index = center_of_gravity(mni_image)

    point_cog_m = masked_image.space.at_index(mask_cog_index)
    point_cog_t = mni_image.space.at_index(template_cog_index)

    offsets = point_cog_m - mask_cog_index @ mni_image.space.basis

    print(f"Mask: {mask_cog_index}")
    print(f"template: {template_cog_index}")
    print(offsets)





    plt.show()
