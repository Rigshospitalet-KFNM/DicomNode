# Python Standard Library
from pathlib import Path
from unittest import skipIf

# Third party modules
import nibabel

# Dicomnode modules
from dicomnode import library_paths
from dicomnode.dicom.series import extract_image
from dicomnode.math.image import Image
from dicomnode.math import CUDA, bounding_box

# Test
from tests.helpers.dicomnode_test_case import DicomnodeTestCase

ct_image = library_paths.report_data_directory / "CT_nifti" / "CT.nii"
ct_brain = library_paths.report_data_directory / "CT_nifti" / "segmentation" / "brain.nii.gz"

files_exists = ct_brain.exists() and ct_brain.exists()

class RegistrationTestCase(DicomnodeTestCase):
  @skipIf(not CUDA and files_exists, "Need GPU and files")
  def test_registration_from_python(self):
    from dicomnode.math import _cuda

    nifti: nibabel.nifti1.Nifti1Image = nibabel.loadsave.load(ct_image) # type: ignore
    image = extract_image(nifti)

    seg_nifti: nibabel.nifti1.Nifti1Image = nibabel.loadsave.load(ct_brain) # type: ignore

    seg_image = extract_image(seg_nifti)

    masked = Image(
      image.raw * seg_image.raw,
      image.space
    )

    print(bounding_box(masked))


    self.assertFalse((masked.raw == image.raw).all())



    _cuda.registration.register(image, masked)
