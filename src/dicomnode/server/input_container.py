# Python standard library
from typing import Any, Dict, List, Optional

# Third party modules
from pydicom import Dataset

# Dicomnode modules
from dicomnode.data_structures.optional import OptionalPath
from dicomnode.dicom.dimse import Address
from dicomnode.lib.io import Directory

class InputContainer:
  """Argument to the processor
  """
  responding_address: Optional[Address]

  def __init__(self,
               data: Dict[str, Any],
               datasets: Dict[str, List[Dataset]] = {},
               paths: OptionalPath = OptionalPath(),
               ) -> None:
    self.__data = data
    self.datasets = datasets
    self.paths = paths
    self.responding_address = None

  def __getitem__(self, key: str):
    if self.__data is None:
      raise KeyError(key)
    return self.__data[key]

  def __contains__(self, key):
    return key in self.__data
