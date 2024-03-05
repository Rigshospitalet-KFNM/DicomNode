"""DEPRECATED MODULE: use dicomnode.data_structures.image_tree instead!"""

from dicomnode.lib.utils import deprecation_message
deprecation_message(__name__, 'dicomnode.data_structures.image_tree')

from dicomnode.data_structures.image_tree import *