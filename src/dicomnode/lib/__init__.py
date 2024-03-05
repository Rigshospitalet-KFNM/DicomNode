# Python standard library
from importlib import import_module

from . import exceptions
from . import logging
from . import io
from . import parser
from . import utils


__all__ = [
  'exceptions',
  'logging',
  'io',
  'utils'
]

deprecated_names = [
  'anonymization',
  'dicom',
  'dicom_factory',
  'dimse',
  'lazy_dataset',
  'image_tree'
  'nifti',
  'numpy_factory',
]

def __dir__():
  return sorted(__all__ + deprecated_names)

# Deprecated imports
__anonymization = None
__dicom = None
__dicom_factory = None
__dimse = None
__image_tree = None
__lazy_dataset = None
__nifti = None
__numpy_factory = None

# Warning should only trigger once, hence they are only called an actual imports
def __getattr__(name):
  if name == 'anonymization':
    global __anonymization
    if __anonymization is None:
      utils.deprecation_message('dicomnode.lib.anonymization',
                                'dicomnode.dicom.anonymization')
      __anonymization = import_module('dicomnode.dicom.anonymization')
      return __anonymization

  if name == 'dicom':
    global __dicom
    if __dicom is None:
      utils.deprecation_message('dicomnode.lib.dicom', 'dicomnode.dicom')
      __dicom = import_module('dicomnode.dicom')
    return __dicom

  if name == 'dicom_factory':
    global __dicom_factory
    if __dicom_factory is None:
      utils.deprecation_message('dicomnode.lib.dicom_factory',
                                'dicomnode.dicom.dicom_factory')
      __dicom_factory = import_module('dicomnode.dicom.dicom_factory')
    return __dicom_factory

  if name == 'dimse':
    global __dimse
    if __dimse is None:
      utils.deprecation_message('dicomnode.lib.dimse',
                                'dicomnode.dicom.dimse')
      __dimse = import_module('dicomnode.dicom.dimse')
    return __dimse

  if name == 'image_tree':
    global __image_tree
    if __image_tree is None:
      utils.deprecation_message('dicomnode.lib.image_tree',
                                'dicomnode.data_structures.image_tree')
      __image_tree = import_module('dicomnode.data_structures.image_tree')
    return __image_tree

  if name == 'lazy_dataset':
    global __lazy_dataset
    if __lazy_dataset is None:
      utils.deprecation_message('dicom.lib.lazy_dataset',
                                'dicomnode.dicom.lazy_dataset')
      __lazy_dataset = import_module('dicomnode.dicom.lazy_dataset')
    return __lazy_dataset

  if name == 'nifti':
    global __nifti
    if __nifti is None:
      utils.deprecation_message('dicomnode.lib.nifti', 'dicomnode.dicom.nifti')
      __nifti = import_module('dicomnode.dicom.nifti')
    return __nifti

  if name == 'numpy_factory':
    global __numpy_factory
    if __numpy_factory is None:
      utils.deprecation_message('dicomnode.lib.numpy_factory',
                                'dicomnode.dicom.numpy_factory')
      __numpy_factory = import_module('dicomnode.dicom.numpy_factory')

  raise AttributeError(f"module {__name__} has no attribute '{name}'")
