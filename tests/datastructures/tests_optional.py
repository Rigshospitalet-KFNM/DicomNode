
# Dicomnode Modules
from dicomnode.lib.exceptions import ContractViolation
from dicomnode.data_structures.optional import OptionalPath

# Test Helpers
from tests.helpers.dicomnode_test_case import DicomnodeTestCase


class OptionalPathTestCases(DicomnodeTestCase):
  def test_invalid_optional_path_raises(self):
    op = OptionalPath()
    self.assertRaises(ContractViolation, getattr, op, 'path')
