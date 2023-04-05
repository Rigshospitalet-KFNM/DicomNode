
# Python standard
from unittest import TestCase

# Dicomnode package
import dicomnode

class InitTestCase(TestCase):
  def test___version__(self):
    self.assertEqual(dicomnode.version(), dicomnode.__version__)
