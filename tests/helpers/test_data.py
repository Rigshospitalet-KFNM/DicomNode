import dicomnode
import nibabel

CT_IMAGE_PATH = dicomnode.library_paths.report_data_directory / "CT_nifti" / "CT.nii"
CT_IMAGE_SEGMENTATION_PATH = dicomnode.library_paths.report_data_directory / "CT_nifti" / "segmentation" / "brain.nii.gz"
MNI_TEMPLATE_PATH = dicomnode.library_paths.report_data_directory / "tpl-MNI152Lin" / "tpl-MNI152Lin_res-02_PD.nii.gz"

class _TestData:
  @property
  def CT_IMAGE(self) -> nibabel.nifti1.Nifti1Image:
    if self._ct_image is None:
      self._ct_image = nibabel.loadsave.load(CT_IMAGE_PATH)

    if not isinstance(self._ct_image, nibabel.nifti1.Nifti1Image):
      raise TypeError("CT Image is not a Nifti 1 Image")

    return self._ct_image

  @property
  def CT_IMAGE_SEGMENTATION(self) -> nibabel.nifti1.Nifti1Image:
    if self._ct_image_segmentation is None:
      self._ct_image_segmentation = nibabel.loadsave.load(CT_IMAGE_SEGMENTATION_PATH)

    if not isinstance(self._ct_image_segmentation, nibabel.nifti1.Nifti1Image):
      raise TypeError("CT Image segmentation is not a Nifti 1 Image")

    return self._ct_image_segmentation

  @property
  def MNI_TEMPLATE(self) -> nibabel.nifti1.Nifti1Image:
    if self._mni_template is None:
      self._mni_template = nibabel.loadsave.load(MNI_TEMPLATE_PATH)

    if not isinstance(self._mni_template, nibabel.nifti1.Nifti1Image):
      raise TypeError("MNI template is not a Nifti 1 Image")

    return self._mni_template

  def __init__(self) -> None:
    self._ct_image = None
    self._ct_image_segmentation = None
    self._mni_template = None

TEST_DATA = _TestData()