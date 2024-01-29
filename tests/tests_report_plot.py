"""_summary_
  """

# Python standard Library
from unittest import TestCase

# Third party Packages
from dicomnode import library_paths
import nibabel

# Dicomnode Packages
from dicomnode.report.plot.selector import AverageSelector, MaxSelector, PercentageSelector
from dicomnode.report.plot.triple_plot import TriplePlot

# Initialization

#nifti_image: nibabel.nifti1.Nifti1Image = nibabel.loadsave.load(f'{library_paths.report_data_directory}/someones_epi.nii.gz') # type: ignore
nifti_image: nibabel.nifti1.Nifti1Image = nibabel.loadsave.load(f'{library_paths.report_data_directory}/someones_anatomy.nii.gz') # type: ignore

class PlotTestCase(TestCase):
  def test_triple_plot(self):
    options = TriplePlot.Options(file_path=f'{library_paths.figure_directory}/triple_plot.png')
    tp = TriplePlot(nifti_image, options)

    tp.save()

  def test_triple_plot_different_selectors(self):
    options = TriplePlot.Options(file_path=f'{library_paths.figure_directory}/different_triple_plot.png',
                                 selector=(PercentageSelector(0.30), MaxSelector(), AverageSelector()))
    tp = TriplePlot(nifti_image, options)

    tp.save()





