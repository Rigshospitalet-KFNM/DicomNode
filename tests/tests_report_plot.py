"""_summary_
  """

# Python standard Library
from unittest import TestCase

# Third party Packages
from dicomnode import library_paths
import nibabel

# Dicomnode Packages
from dicomnode.report.plot import TriplePlot

# Initialization

#nifti_image: nibabel.nifti1.Nifti1Image = nibabel.loadsave.load(f'{library_paths.report_data_directory}/someones_epi.nii.gz') # type: ignore
nifti_image: nibabel.nifti1.Nifti1Image = nibabel.loadsave.load(f'{library_paths.report_data_directory}/someones_anatomy.nii.gz') # type: ignore

class PlotTestCase(TestCase):
  def test_triple_plot(self):
    tp = TriplePlot(f'{library_paths.figure_directory}/triple_plot.png', nifti_image)

    tp.save()





