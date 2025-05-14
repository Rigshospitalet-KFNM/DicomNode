# Python Standard library
from pathlib import Path
from typing import List
from unittest import skipIf

# Third party packages
import numpy
from pydicom import Dataset, dcmread
import nibabel
from nibabel import nifti1

import matplotlib
matplotlib.use('tkAgg')

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
# Dicomnode Modules
from dicomnode.dicom import series
from dicomnode.dicom.nifti import convert_to_nifti
from dicomnode.math import transpose_nifti_coords
from dicomnode.math.image import Image

from tests.helpers.dicomnode_test_case import DicomnodeTestCase
from tests.helpers import generate_numpy_datasets

_REPORT_DATA_PATH =Path('report_data')
_CT_IMAGE_PATH = _REPORT_DATA_PATH / 'CT'
_PATHS = [p.absolute() for p in _CT_IMAGE_PATH.glob('*.dcm')]

CT_IMAGE_EXISTS = _CT_IMAGE_PATH.exists() and len(_PATHS) > 10

NIFTI_PATH = (_REPORT_DATA_PATH / 'CT_nifti' / 'CT.nii').absolute()

class DicomnodeDicomNifti(DicomnodeTestCase):
  def test_can_nifti(self):
    datasets = series.DicomSeries([ds for ds in generate_numpy_datasets(10, Rows=10, Cols=10)])

    datasets["Modality"] = "CT"

    nifti = convert_to_nifti(datasets.datasets, None, False)
    self.assertIsInstance(nifti, nifti1.Nifti1Image)

  @skipIf(not CT_IMAGE_EXISTS, "Need a ct image in report_data/CT if this test need to work")
  def test_nifti_and_my_data_are_the_asdf(self):
    ct_datasets: List[Dataset] = [dcmread(p) for p in _PATHS] #type: ignore
    self.assertGreater(len(ct_datasets), 0)

    nifti: nifti1.Nifti1Image = nibabel.load(NIFTI_PATH) #type: ignore
    nifti_data = nifti.get_fdata(dtype='float32')

    ct_image = Image.from_datasets(ct_datasets)

    raw_ct_image = ct_image.raw
    self.assertEqual(nifti.shape, tuple(reversed(raw_ct_image.shape)))

    switched = transpose_nifti_coords(raw_ct_image)

    self.assertEqual(switched.dtype, nifti_data.dtype)
    self.assertTrue((switched == nifti_data).all())
