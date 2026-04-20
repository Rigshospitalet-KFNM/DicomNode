# Python standard library
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Generator, Optional, Type

# Third party
from pydicom import Dataset

# Dicomnode modules
from dicomnode.config import DicomnodeConfig
from dicomnode.data_structures.optional import OptionalPath
from dicomnode.lib.exceptions import IncorrectlyConfigured, ContractViolation
from dicomnode.lib.io import Directory, File, save_dicom, load_dicoms

class Storage(ABC):
  """A interface class for storing pydicom datasets with the following methods:

  * add_image
  * __iter__
  * __len__
  * __contains__
  """

  def __init__(self, storage_location: OptionalPath) -> None:
    super().__init__()

  @abstractmethod
  def store_image(self, dataset : Dataset): #pragma: no cover
    raise NotImplemented

  @abstractmethod
  def __iter__(self) -> Generator[Dataset, None, None]: #pragma: no cover
    raise NotImplemented

  @abstractmethod
  def __contains__(self, dataset: Dataset): #pragma: no cover
    raise NotImplemented

  @abstractmethod
  def __len__(self): # pragma: no cover
    raise NotImplemented


class InMemoryStorage(Storage):
  """Storage of dataset in memory without any files

    Args:
      name: The class name of the input that contains this storage
      config: The config used to create
  """

  def __init__(self, storage_location: OptionalPath) -> None:
    self._storage: Dict[str, Dataset] = {}

  def store_image(self, dataset):
    self._storage[dataset.SOPInstanceUID] = dataset

  def __iter__(self):
    yield from self._storage.values()

  def __contains__(self, dataset: Dataset):
    return dataset.SOPInstanceUID in self._storage

  def __len__(self):
    return len(self._storage)


class LazyFileStorage(Storage):
  def __init__(self, storage_location: OptionalPath) -> None:
    self._storage_location: Path = storage_location.path

  def store_image(self, dataset: Dataset):
    path = self.path_for_dataset(dataset)
    save_dicom(path, dataset)

  def __iter__(self) -> Generator[Dataset, None, None]:
    if self._storage_location is None:
      raise StopIteration
    yield from load_dicoms(self._storage_location)

  def path_for_dataset(self, dataset: Dataset) -> Path:

    if self._storage_location is None:
      raise ContractViolation("You set path without ")

    if 0x0008103E in dataset and 0x0020_0013 in dataset:
      return self._storage_location / f"{dataset.SeriesDescription}_{dataset.InstanceNumber}.dcm"

    return self._storage_location / (str(dataset.SOPInstanceUID) + ".dcm")

  def __contains__(self, dataset: Dataset):
    path = self.path_for_dataset(dataset)
    return path.exists()

  def __len__(self):
    if self._storage_location is None:
      return 0
    items = 0
    for item in self._storage_location.glob('*.dcm'):
      items += 1

    return items


class FileStorage(LazyFileStorage): # I could see an argument for just inheriting from Storage
  def __init__(self, storage_location: OptionalPath) -> None:
    super().__init__(storage_location)

    self._storage = {}
    # Loading from archive is done initialization from dicomnode not from the storage


  def store_image(self, dataset: Dataset):
    self._storage[dataset.SOPInstanceUID] = dataset

    return super().store_image(dataset)

  def __iter__(self) -> Generator[Dataset, None, None]:
    yield from self._storage.values()

  def __contains__(self, dataset: Dataset):
    return dataset.SOPInstanceUID in self._storage

  def __len__(self):
    return len(self._storage)


def get_storage_from_config(config: DicomnodeConfig) -> Type[Storage]:
  if config.ARCHIVE_DIRECTORY:
    if config.LAZY_STORAGE:
      return LazyFileStorage
    else:
      return FileStorage

  return InMemoryStorage