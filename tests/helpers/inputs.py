"""This module contains various implementations of Abstract inputs used in
various test
"""

# Python3 Standard Library
from typing import List

# Third party libraries
from pydicom import Dataset

# Dicomnode
from dicomnode.dicom.dimse import Address
from dicomnode.server.input import AbstractInput, HistoricAbstractInput
from dicomnode.server.grinders import ListGrinder

# Test helpers
from tests.helpers.storage_endpoint import ENDPOINT_PORT

class TestInput(AbstractInput):
  required_tags: List[int| str] = [0x00080018, 0x00100040]

  def validate(self):
    return True

class NeverValidatingInput(AbstractInput):
  required_tags: List[int | str] = [0x00080018]

  def validate(self):
    return False

class TestHistoricInput(HistoricAbstractInput):
  address = Address('localhost', ENDPOINT_PORT, "DUMMY")
  required_tags: List[int | str] = [0x00080018]



class ListInput(AbstractInput):
  required_tags = [0x0008_0018]

  image_grinder = ListGrinder()

  def validate(self) -> bool:
    return True