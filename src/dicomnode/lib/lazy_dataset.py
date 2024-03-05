"""DEPRECATED MODULE: use dicomnode.dicom.lazy_dataset instead!"""

from dicomnode.lib.utils import deprecation_message
deprecation_message(__name__, 'dicomnode.dicom.lazy_dataset')

from dicomnode.dicom.lazy_dataset import *