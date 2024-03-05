"""DEPRECATED MODULE: use dicomnode.dicom.anonymization instead!"""

from dicomnode.lib.utils import deprecation_message
deprecation_message(__name__, 'dicomnode.dicom')

from dicomnode.dicom import *