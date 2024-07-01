"""This module contains various "validators" these are used in AbstractInputs
for the `required_values` attribute. Sometimes the good old `==` operator is
insufficient and these validators exists to expand functionality allowing you
to overload this:

For example if you want an input accept two different SOP classes.

class Input(AbstractInput):
  required_values = {
    0x0008_0016 : OptionsValidator([
      SOP_Class_Name_1,
      SOP_Class_Name_2
    ])
  }

Note: maybe this is a bad name because they are used in AbstractInput which have
a method called validate, which these have NOTHING to with required_values
"""

# Python Standard Library
from re import Pattern
from typing import Any,Iterable, Optional, Union

# Third party Modules

# Dicomnode Modules
from dicomnode.lib.regex import from_wildcard

class Validator:
  """Interface class for input Validators"""
  def __init__(self, value: Any) -> None:
    pass

  def __call__(self, target: Any) -> bool:
    return False
  

class EqualityValidator(Validator):
  def __init__(self, value: Any) -> None:
    super().__init__(value)
    self.value = value

  def __call__(self, target: Any) -> bool:
    return self.value == target


class RegexValidator(Validator):
  def __init__(self, value: Union[str, Pattern]) -> None:
    super().__init__(value)
    if isinstance(value, str):
      self.pattern = from_wildcard(value)
    else:
      self.pattern = value

  def __call__(self, target: Any) -> bool:
    return  self.pattern.match(target) is not None


class OptionsValidator(Validator):
  def __init__(self, value: Iterable[Any], validator: Optional[Validator] = None) -> None:
    super().__init__(value)
    self.options = value
    self.optional_validator = validator

  def __call__(self, target: Any) -> bool:
    return_value = False
    for value in self.options:
      if self.optional_validator is None:
        return_value |= value == target
      else:
        return_value |= self.optional_validator(target)
    return value


def get_validator_for_value(value):
  if isinstance(value, Validator):
    return value
  if isinstance(value, Pattern):
    return RegexValidator(value)
  
  return EqualityValidator(value)
