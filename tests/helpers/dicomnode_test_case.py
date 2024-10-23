from re import compile,Pattern
from typing import List, Union
from pprint import pformat
from unittest import TestCase
from unittest.util import safe_repr

from dicomnode.lib.regex import escape_pattern

class DicomnodeTestCase(TestCase):
  def assertRegexIn(self, regex: Union[Pattern, str], container: List[str], msg=None):
    if isinstance(regex, str):
      regex = escape_pattern(regex)

    found_pattern = False
    for str_ in container:
      found_pattern |= regex.search(str_) is not None

    if not found_pattern:
      msg = self._formatMessage(msg, f"Pattern {safe_repr(regex.pattern)} is not in {pformat(container)}")
      self.fail(msg)
