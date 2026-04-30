# Python standard library
from typing import Any, Dict, List, Optional

# Third party modules
from pydicom import Dataset

# Dicomnode modules
from dicomnode.dicom.dimse import Address
from dicomnode.data_structures.optional import OptionalPath

class InputContainer:
  """Argument to the processor - it's a plain data object.
     * data contains the grounded data
     * datasets contains the datasets used to produce the data.
     * paths is the optional directory to input that contains the images - so if
     you have images and the input is named CT you can find the CT's images at
     paths / "CT"
  """

  def __init__(self,
               data: Dict[str, Any],
               datasets: Dict[str, List[Dataset]] = {},
               paths: OptionalPath = OptionalPath(),
               ) -> None:
    self.__data = data
    self.datasets = datasets
    self.paths = paths
    self.responding_address: Optional[Address] = None

  def __getitem__(self, key: str):
    return self.__data[key]

  def __contains__(self, key):
    return key in self.__data
