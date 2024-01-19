"""Test cases for generating latex report

In general the test case will be checked on the generated LaTeX code.
"""

# Python3 Standard Library imports
from unittest import TestCase

# Third party imports

# Dicomnode Imports
from dicomnode.report import generator

class GeneratorTestCase(TestCase):
  def test_empty_report(self):

    test_file_path = "test_file.text"

    report = generator.Report(test_file_path)

    report.generate_tex()

    with open(test_file_path,'r') as fp:
      print(fp.readlines())


