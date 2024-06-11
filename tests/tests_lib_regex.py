# Python standard library
from unittest import TestCase

# Third module

# Dicomnode modules
from dicomnode.lib.regex import from_wildcard

class RegexTestCases(TestCase):
  def test_regex_use_case(self):
    file_str = "dicomnode/*"

    regex = from_wildcard(file_str)

    self.assertIsNotNone(regex.match("dicomnode/asdfasdfasdf"))
    self.assertIsNotNone(regex.match("dicomnode/hello_world"))
    self.assertIsNone(regex.match("icomnode/hello_world"))
