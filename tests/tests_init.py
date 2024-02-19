# Python standard
from importlib import import_module
from unittest import TestCase, main

# Dicomnode package
import dicomnode

class InitTestCase(TestCase):
  def test_version(self):
    self.assertEqual(dicomnode.version(), dicomnode.__version__)

  def test_lazy_imports(self):
    self.assertIs(dicomnode.constants, import_module('dicomnode.constants'))
    self.assertIs(dicomnode.lib, import_module('dicomnode.lib'))
    self.assertIs(dicomnode.server, import_module('dicomnode.server'))
    self.assertIs(dicomnode.report, import_module('dicomnode.report'))
    self.assertIs(dicomnode.tools, import_module('dicomnode.tools'))

if __name__ == '__main__':
  main()