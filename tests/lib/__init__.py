from tests.helpers.dicomnode_test_case import DicomnodeTestCase

from . import tests_validators
from dicomnode import lib


class libTestCase(DicomnodeTestCase):
  def test_sorted(self):
    prev = None
    for name in dir(lib):
      if prev is not None:
        self.assertLess(prev, name)
      prev = name
