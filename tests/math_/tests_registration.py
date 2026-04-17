# Python Standard Library
from pathlib import Path
from unittest import skipIf, skip

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
from tests.helpers import test_data
from tests.helpers.dicomnode_test_case import DicomnodeTestCase

ct_image_path = library_paths.report_data_directory / "CT_nifti" / "CT.nii"
ct_brain_path = library_paths.report_data_directory / "CT_nifti" / "segmentation" / "brain.nii.gz"
mniBrain_path = library_paths.report_data_directory / "tpl-MNI152Lin" / "tpl-MNI152Lin_res-02_PD.nii.gz"

files_exists = ct_brain_path.exists() and ct_brain_path.exists() and mniBrain_path.exists()

class RegistrationTestCase(DicomnodeTestCase):
  @skip("Registration is not done yet")
  @skipIf((not CUDA) or (not test_data.USING_TEST_DATA), "Need GPU and files")
  def test_registration_from_python(self):

    nifti = test_data.TEST_DATA.CT_IMAGE
    image = extract_image(nifti)

    seg_nifti = test_data.TEST_DATA.CT_IMAGE_SEGMENTATION

    seg_image = extract_image(seg_nifti)

    mni = test_data.TEST_DATA.MNI_TEMPLATE
    mni_image = extract_image(mni)

    masked_image = mask_image(image, seg_image)

    mask_cog_index = center_of_gravity(masked_image)
    template_cog_index = center_of_gravity(mni_image)

    point_cog_m = masked_image.space.at_index(mask_cog_index)
    point_cog_t = mni_image.space.at_index(template_cog_index)

    offsets = point_cog_m - mask_cog_index @ mni_image.space.basis

    print(f"Mask: {mask_cog_index}")
    print(f"template: {template_cog_index}")
    print(f"offsets: {offsets}")


    plt.show()
