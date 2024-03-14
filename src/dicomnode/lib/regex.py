from re import compile, Pattern

def from_wildcard(string: str) -> Pattern:
  """Builds a regex from a wildcard

  Args:
      string (str): a string you wish convert to a regex, don't put a normal regex in here

  Returns:
      Pattern: _description_
  """
  return compile(string.replace('*', '.*'))