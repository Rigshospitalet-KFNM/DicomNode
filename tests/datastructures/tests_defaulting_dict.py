# Python standard library
from typing import cast, List, Set

# Third party modules

# Dicomnode packages
from dicomnode.lib.exceptions import ContractViolation
from dicomnode.data_structures.defaulting_dict import DefaultingDict

# Testing
from tests.helpers.dicomnode_test_case import DicomnodeTestCase

class DefaultingDictTestCases(DicomnodeTestCase):
  """These mostly show error cases, as functionality is demonstrated by other
  tests"""
  def test_defaulting_dict_success_on_build_ins(self):
    defaulting_dict: DefaultingDict[int, List[str]] = cast(DefaultingDict[int, List[str]],DefaultingDict(list))
    defaulting_dict[1].append("Hello world")

    defaulting_dict_set: DefaultingDict[int, Set[str]] = cast(DefaultingDict[int, Set[str]], DefaultingDict(set))
    defaulting_dict_set[1].add("Hello world")


  def test_defaulting_dict_errors_function_with_two_arguments(self):
    self.assertRaises(ContractViolation, DefaultingDict, lambda x,y : x)