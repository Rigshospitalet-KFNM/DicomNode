"""YES THESE TESTS ARE JUST FOR COVERAGE!"""
# Python standard library

# Third party modules

# Dicomnode modules
from dicomnode.server.input_container import InputContainer

# Test modules
from tests.helpers.dicomnode_test_case import DicomnodeTestCase

class InputContainerTestCases(DicomnodeTestCase):
  def tests_input_containers_contains(self):
    ic = InputContainer({ 'key' : None})

    self.assertFalse("not key" in ic)
    self.assertTrue("key" in ic)
    self.assertIsNone(ic["key"])
