from re import compile, Pattern

def escape_pattern(string: str) -> Pattern:
  escaped_string = string.replace("[", r"\[").replace("]", r"\]")

  return compile(escaped_string)


def from_wildcard(string: str) -> Pattern:
  """Builds a regex from a wildcard

  Args:
      string (str): a string you wish convert to a regex, don't put a normal regex in here

  Returns:
      Pattern: _description_
  """
  return compile(string.replace('*', '.*'))

__all__ = [
  'from_wildcard'
]
