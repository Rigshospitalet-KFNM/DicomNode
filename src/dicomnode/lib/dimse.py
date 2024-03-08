"""DEPRECATED MODULE: use dicomnode.dicom.anonymization instead!"""

from dicomnode.lib.utils import deprecation_message
deprecation_message(__name__, 'dicomnode.dicom.dimse')

from dicomnode.dicom.dimse import *