# Python Standard library
import re
from unittest import TestCase

# Third party modules

# Dicomnode modules
from dicomnode.lib.validators import EqualityValidator, RegexValidator,\
  OptionsValidator, get_validator_for_value, NegatedValidator,\
  CaselessRegexValidator

class ValidatorsTestCases(TestCase):
  def test_validator(self):
    eq_validator = EqualityValidator(1)

    self.assertTrue(eq_validator(1))
    self.assertFalse(eq_validator(2))
    self.assertFalse(eq_validator("asdfasdfasdf"))

  def test_regex_validator(self):
    regex_validator = RegexValidator("PET")

    self.assertTrue(regex_validator("This string contains PET"))
    self.assertFalse(regex_validator("This string contains pet"))

    regex_validator = RegexValidator(re.compile(r"^\d+$"))

    self.assertTrue(regex_validator("123541235"))
    self.assertFalse(regex_validator("12354Hello1235"))

  def test_options_validator(self):
    options_validator = OptionsValidator([1,3,5])

    self.assertTrue(options_validator(1))
    self.assertFalse(options_validator(2))
    self.assertTrue(options_validator(3))
    self.assertFalse(options_validator(4))
    self.assertTrue(options_validator(5))
    self.assertFalse(options_validator(6))
    self.assertFalse(options_validator(7))

    options_validator = OptionsValidator(["PET", "CT"], RegexValidator)

    self.assertTrue(options_validator("asdfasdfPETasdfasf"))
    self.assertTrue(options_validator("asdfasdfCTasdfasf"))
    self.assertFalse(options_validator("asdfasdfSPECasdfasf"))

  def test_get_validator_for_value(self):
    validator = get_validator_for_value("123")

    self.assertIsInstance(validator,EqualityValidator)
    self.assertTrue(validator("123"))
    self.assertFalse(validator(123))

    self.assertIs(get_validator_for_value(validator), validator)

    validator = get_validator_for_value(re.compile(r"\d+"))

  def test_negated_validator(self):
    validator = NegatedValidator(
      RegexValidator(
        "TEST"
      )
    )

    self.assertFalse(validator("TEST"))
    self.assertTrue(validator("test"))

  def test_caseless_regex(self):
    validator = CaselessRegexValidator("SpOnGeBoB")

    self.assertTrue(validator("spongebob"))
    self.assertTrue(validator("sPoNgEbOb"))
    self.assertTrue(validator("SPONGEBOB"))
    self.assertTrue(validator("SPONGEBOB is in here"))
    self.assertTrue(validator("asdf SPONGEBOB is in here"))

    self.assertFalse(validator("squidward"))
    self.assertFalse(validator("SQUIDWARD"))

  def test_caseless_regex_negated(self):
    validator = NegatedValidator(CaselessRegexValidator("SpOnGeBoB"))

    self.assertFalse(validator("spongebob"))
    self.assertFalse(validator("sPoNgEbOb"))
    self.assertFalse(validator("SPONGEBOB"))
    self.assertFalse(validator("SPONGEBOB is in here"))
    self.assertFalse(validator("asdf SPONGEBOB is in here"))
    self.assertTrue(validator("squidward"))
    self.assertTrue(validator("SQUIDWARD"))
