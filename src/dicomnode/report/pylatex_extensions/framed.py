""""""

__author__ = "Christoffer Vilstrup Jensen"

# Python Standard Library

# Third Party Packages
from pylatex import Package
from pylatex.base_classes import Environment

# Dicomnode packages

class Framed(Environment):
  packages = [Package('framed'), Package('xcolor')]
