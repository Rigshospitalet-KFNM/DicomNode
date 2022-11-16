"""This module concern itself with defining user input. In here there's a
number of classes which you should use to define your input for your process function.


"""

from abc import abstractmethod, ABC
from pathlib import Path
from pydicom import Dataset
from typing import List, Callable, Dict, Tuple, Any, Optional, Iterator, TypeVar


from dicomnode.lib.exceptions import InvalidDataset
from dicomnode.lib.io import load_dicom, save_dicom
from dicomnode.lib.grinders import identity_grinder
from dicomnode.lib.imageTree import ImageTreeInterface
from dicomnode.lib.utils import staticfy


GrindType = TypeVar('GrindType')

class AbstractInput(ImageTreeInterface, ABC):
  required_tags: List[int] = [0x00080018, 0x7FE00010] # InstanceUID, Pixel Data
  private_tags: Dict[int, Tuple[str, str, str, str, str]] = {}
  required_values: Dict[int, Any] = {}
  image_grinder: Callable[[Iterator[Dataset]], GrindType] = identity_grinder

  @abstractmethod
  def validate(self) -> bool:
    """Checks if the input have sufficient data, to start processing

    Returns:
        bool: If there's sufficient data to start processing
    """
    return False

  def get_data(self) -> GrindType:
    """This function retrieves all the data stores in the input,
    and makes it ready for processing

    Returns:
        Any: Data ready for the pipelines process function.
    """
    return staticfy(self.image_grinder)(self.data.values())

  def __init__(self, instance_directory: Optional[Path] = None, in_memory: bool = False):
    self.__instance_directory: Path = instance_directory
    self.in_memory = in_memory
    self.data: Dict[str, Dataset] = {}
    self.images = 0

    if 0x00080018 not in self.required_tags: # Tag for SOPInstance is (0x0008,0018)
      self.required_tags.push(0x00080018)

    if not self.in_memory:
      for image_path in self.__instance_directory.iterdir():
        dcm = load_dicom(image_path, self.private_tags)
        self.add_image(dcm)


  def map(self, func: Callable[[Dataset], Any], UIDMapping) -> List[Any]:
    pass


  def __getPath(self, dicom: Dataset) -> Path:
    """Gets the path, where a dataset would be saved.

    Args:
        dicom (Dataset): The dataset in question

    Returns:
        Path: The path for that dataset.
    """
    image_name: str = ""
    if 0x00080060 in dicom: # Modality
      image_name += f"{dicom.Modality}_"
    image_name += "image"

    if 0x00200013 in dicom: # Instance Number
      image_name += f"_{dicom.InstanceNumber}"
    else:
      image_name += f"_{dicom.SOPInstanceUID.name}"

    image_name += ".dcm"

    return self.__instance_directory / image_name

  def add_image(self, dicom: Dataset) -> None:
    """Attempts to add an image to the input.

    Args:
        dicom (Dataset): The dataset to be added

    Raises:
        InvalidDataset: If the dataset is not valid, this is raised.
    """
    # Dataset Validation
    for required_tag in self.required_tags:
      if required_tag not in dicom:
        raise InvalidDataset()

    for required_tag, required_value in self.required_values.items():
      if required_tag not in dicom:
        raise InvalidDataset()
      if dicom[required_tag] != required_value:
        raise InvalidDataset()

    # Save the dataset
    self.data[dicom.SOPInstanceUID.name] = dicom # Tag for SOPInstance is (0x0008,0018)
    self.images += 1
    if not self.in_memory:
      dicom_path:Path = self.__getPath(dicom)
      if not dicom_path.exists():
        save_dicom(dicom_path, dicom)