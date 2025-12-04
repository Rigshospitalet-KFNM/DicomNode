# Python standard
from importlib import import_module
from unittest import TestCase, main

# Dicomnode package
import dicomnode

from tests.helpers.dicomnode_test_case import DicomnodeTestCase

class InitTestCase(DicomnodeTestCase):
  def test_version(self):
    self.assertEqual(dicomnode.version(), dicomnode.__version__)

  def test_lazy_imports(self):
    self.assertIs(dicomnode.constants, import_module('dicomnode.constants'))
    self.assertIs(dicomnode.lib, getattr(dicomnode, 'lib'))
    self.assertIs(dicomnode.lib,       import_module('dicomnode.lib'))
    self.assertIs(dicomnode.server,    import_module('dicomnode.server'))
    self.assertIs(dicomnode.report,    import_module('dicomnode.report'))
    self.assertIs(dicomnode.tools,     import_module('dicomnode.tools'))

    self.assertRaises(AttributeError, getattr, dicomnode, 'NotAModule')

  def test_dir(self):
    self.assertEqual(dir(dicomnode),[
      'constants',
      'data_structures',
      'dicom',
      'lib',
      'library_paths',
      'math',
      'report',
      'server',
      'tools',
      'version',
    ])

if __name__ == '__main__':
  main()