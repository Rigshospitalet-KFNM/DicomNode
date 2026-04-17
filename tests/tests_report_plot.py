"""_summary_
  """

# Python standard Library
from pathlib import Path
from typing import Optional
from unittest import TestCase, skipIf

# Third party Packages
from dicomnode import library_paths
import nibabel

# Dicomnode Packages
from dicomnode.lib.io import load_dicoms
from dicomnode.report.plot.selector import AverageSelector, MaxSelector, PercentageSelector
from dicomnode.report.plot.triple_plot import TriplePlot

from tests.helpers import test_data
from tests.helpers.dicomnode_test_case import DicomnodeTestCase


# Path to images
nifti_path = Path(f'{library_paths.report_data_directory}/someones_anatomy.nii.gz')
ct_path = Path(f'{library_paths.report_data_directory}/CT')


class PlotTestCase(DicomnodeTestCase):
  @skipIf(not test_data.USING_TEST_DATA, "Needs nifti data to plot")
  def test_triple_plot(self):

    options = TriplePlot.Options(file_path=f'{library_paths.figure_directory}/triple_plot.png')
    tp = TriplePlot(test_data.TEST_DATA.CT_IMAGE, options=options) # type: ignore

    tp.save()

  @skipIf(not test_data.USING_TEST_DATA, "Needs nifti data to plot")
  def test_triple_plot_different_selectors(self):
    options = TriplePlot.Options(file_path=f'{library_paths.figure_directory}/different_triple_plot.png',
                                 selector=(PercentageSelector(0.30), MaxSelector(), AverageSelector()))
    tp = TriplePlot(test_data.TEST_DATA.CT_IMAGE, options=options)# type: ignore

    tp.save()

  @skipIf(not ct_path.exists(), "Needs CT data")
  def test_triple_plot_dicom_data(self):
    datasets = load_dicoms(ct_path)

    options = TriplePlot.Options(file_path=f'{library_paths.figure_directory}/ct_triple_plot.png',
                                 selector=(PercentageSelector(0.30), MaxSelector(), AverageSelector()))
    tp = TriplePlot(datasets, options=options)

    tp.save()
