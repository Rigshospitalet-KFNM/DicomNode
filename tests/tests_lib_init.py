"""This file is mostly here to check that a warning is raises on deprecated
imports"""

# Python Standard library
from unittest import TestCase

# Third party packages

# Dicomnode packages


class DeprecationWarningTests(TestCase):
  def test_lib_nifti(self):
    with self.assertWarns(DeprecationWarning):
      from dicomnode.lib.nifti import convert_to_nifti

  def test_lib_lazy_dataset(self):
    with self.assertWarns(DeprecationWarning):
      from dicomnode.lib.lazy_dataset import LazyDataset

  def test_lib_image_tree(self):
    with self.assertWarns(DeprecationWarning):
      from dicomnode.lib.image_tree import ImageTreeInterface

  def test_lib_anonymization(self):
    with self.assertWarns(DeprecationWarning):
      from dicomnode.lib.anonymization import anonymize_dataset
