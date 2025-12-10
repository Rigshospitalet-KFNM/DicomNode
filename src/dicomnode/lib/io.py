""""""

__author__ = "Christoffer Vilstrup Jensen"

# Python Standard Library
from enum import Enum
import errno
import fcntl
from logging import Logger
import os
from pathlib import Path
import random
import time
import shutil
from typing import Dict, List, Optional, Tuple, Type, TypeVar, Union

# Thrid party Packages
import pydicom
from pydicom.filewriter import dcmwrite
from pydicom import Dataset, Sequence
from pydicom.errors import InvalidDicomError
from pydicom.values import convert_SQ, convert_string
from pydicom.datadict import DicomDictionary, keyword_dict #type: ignore Yeah Pydicom have some fancy import stuff.

# Dicomnode Library
from dicomnode.lib.utils import type_corrosion

def update_private_tags(new_dict_items : Dict[int, Tuple[str, str, str, str, str]]) -> None:
  """Updated the dicom dictionary with a set of new private tags,
  allowing pydicom to reconize private tags.

  Args:
    new_dict_items (dict[int, Tuple[str, str, str, str, str]])

  Example:
  >>> try:
  >>>   dicomObject.NewPrivateTag
  >>> except AttributeError:
  >>>   print("This happens")
  >>> update_private_tags({0x13374201 : ('LO', '1', 'New Private Tag', '', 'NewPrivateTag')})

  """
  # Update DicomDictionary to include our private tags
  DicomDictionary.update(new_dict_items)
  new_names_dict = dict([(val[4], tag) for tag, val in new_dict_items.items()])
  keyword_dict.update(new_names_dict)


def apply_private_tags(
    dataset: Dataset,
    private_tags: Dict[int, Tuple[str, str, str, str, str]] = {},
    is_implicit_VR:bool = True,
    is_little_endian:bool = True
  ):
  """_summary_

  Args:
      dataset (Dataset): Dataset to be updated
      private_tags (Dict[int, Tuple[str, str, str, str, str]], optional): Private tags to update the dataset with. Defaults to {}.
      is_implicit_VR (bool, optional): encoding of dicom. Defaults to True.
      is_little_endian (bool, optional): Byte Encoding of dicom image. Defaults to True.

  Returns:
      Dataset: Modified dataset with updated index

  Example:
  >>> ds = Dataset()
  >>> ds.add_new(0x13374269, 'UN', self.test_string.encode())
  >>> io.apply_private_tags(ds, {0x13374269 : ("LO", "1", "New Private Tag", "", "NewPrivateTag")})
  >>> ds
  (1337, 4269) Private tag data                    LO: 'hello world'

  """
  for data_element in dataset:
    if data_element.tag not in private_tags and data_element.VR != 'SQ':
      continue

    if data_element.VR == 'UN':
      if private_tags[data_element.tag][0] == 'SQ':
        ds_sq = convert_SQ(data_element.value, is_implicit_VR , is_little_endian)
        seq_list = []
        for sq in ds_sq:
          sq = apply_private_tags(sq, private_tags=private_tags, is_little_endian=is_little_endian, is_implicit_VR=is_implicit_VR)
          seq_list.append(sq)
        dataset.add_new(data_element.tag, private_tags[data_element.tag][0], Sequence(seq_list))
      elif private_tags[data_element.tag][0] == 'LO':
        new_val = convert_string(data_element.value, is_little_endian)
        dataset.add_new(data_element.tag, private_tags[data_element.tag][0], new_val)
      else:
        dataset.add_new(data_element.tag, private_tags[data_element.tag][0], data_element.value)
    elif data_element.VR == 'SQ':
      for ds_sq in data_element:
        apply_private_tags(ds_sq, private_tags=private_tags, is_little_endian=is_little_endian, is_implicit_VR=is_implicit_VR)
  return dataset


@type_corrosion(Path)
def discover_files(source: Path) -> List[Path]:
  discover_stack: List[Path] = [source]
  files: List[Path] = []

  while discover_stack:
    path = discover_stack.pop()
    if path.is_file():
      files.append(path)
    if path.is_dir():
      for subpath in path.glob('*'):
        if len(subpath.name) and subpath.name[0] != '.':
          discover_stack.append(subpath)
  return files

@type_corrosion(Path)
def load_dicom(dicom_path: Path) -> Dataset:
  """Loads a single dicom image, doesn't parse private tags
  To parse multiple dataset use load_dicoms

  Args:
      dicom_path (Path): Path to a dicom

  Raises:
      FileNotFoundError: when dicom_path doesn't exists
      pydicom.errors.InvalidDicomError: Raised when a path doesn't point to a
      dicom object

  Returns:
      Dataset, List[Dataset]]: _description_
  """
  if not dicom_path.exists():
    raise FileNotFoundError("File doesn't exists")
  if dicom_path.is_dir():
    raise IsADirectoryError("Loading from a directory is not supported, use load_dicoms")

  return pydicom.dcmread(dicom_path)

@type_corrosion(Path)
def load_dicoms(dicom_path: Path) -> List[Dataset]:
  if not dicom_path.exists():
    raise FileNotFoundError
  datasets = []
  if dicom_path.is_file():
    try:
      dataset = pydicom.dcmread(dicom_path)
      datasets.append(dataset)
    except InvalidDicomError:
      pass

  if dicom_path.is_dir():
    for path in dicom_path.glob('*'):
      if path.is_dir() and not (path.name == '..' or path.name == ''):
        dir_datasets = load_dicoms(path)
        datasets += dir_datasets
      else:
        try:
          dataset = pydicom.dcmread(path)
          datasets.append(dataset)
        except InvalidDicomError:
          continue

  def sorting_function(ds):
    if 0x00200013 in ds:
      return ds[0x00200013].value
    return -1

  datasets.sort(key=sorting_function)

  return datasets

def save_dicom(
    dicom_path: Path,
    dicom: Dataset,
    overwrite = True
  ):
  """A method similar to pydicom.dcmwrite, with sane defaults, and creates the
  directory that the dicom is in, so you don't have to ensure a directory
  structure is there.

  Args:
      dicom_path (Path): Path to where you wish to store the dataset
      dicom (Dataset): The dataset to be saved
      overwrite(boolean) : If false, then this function will error when
                           overwritting a file
  """
  dicom_path = dicom_path.absolute()
  if not dicom_path.parent.exists():
    dicom_path.parent.mkdir(parents=True)

  dicom.save_as(dicom_path, enforce_file_format=True, overwrite=overwrite)

class TemporaryWorkingDirectory():
  """Creating a temporary directory for work to be done in

  Args:
    path_to_new_dir (Path | str) - path to the directory to be created

  Example:
  basic use-case

  >>>with TemporaryWorkingDirectory("/tmp/dir") as dir:
  >>>  os.getcwd()
  /tmp/dir
  >>>os.getcwd()
  current/working/directory

  """

  def __init__(self, path_to_new_dir: Union[Path,str]) -> None:
    if isinstance(path_to_new_dir, str):
      path_to_new_dir = Path(path_to_new_dir)
    self.__cwd = os.getcwd()
    self.temp_directory_path = path_to_new_dir
    if not path_to_new_dir.exists():
      path_to_new_dir.mkdir(exist_ok=True, parents=True)

  def __enter__(self):
    os.chdir(self.temp_directory_path)
    return self

  def __exit__(self, exc_type: Type[Exception], exc_val: Exception, exc_tb):
    os.chdir(self.__cwd)
    if self.temp_directory_path.exists():
      shutil.rmtree(self.temp_directory_path)

class DicomLazyIterator:
  """This class is for creating a lazy dataset iterator for large datasets
  """
  def __init__(self, path: Path, dicom_fileheader="dcm") -> None:
    self._path = path
    self._dicom_fileheader = dicom_fileheader

  def __iter__(self):
    for sub_path in self._path.glob(f'**/*.{self._dicom_fileheader}'):
      yield load_dicom(sub_path)

class ResourceFile:
  """A Resource file is a queue of processes each waiting to gain access to a
  resource. The absence of this file indicate that the resource is available.

  This file doesn't handle any acquisition or release of the resource
  """

  def __init__(self,
               logger : Logger,
               file_path: str | Path,
               ):
    self.resource = file_path
    self.pid = os.getpid()
    self.logger = logger
    self.locked = False


  def __enter__(self):
    while True:
      retries = 0
      try:
        self.resource_file = open(self.resource, 'r')
        fcntl.flock(self.resource_file, fcntl.LOCK_EX)
        self.logger.info(f"Process: {self.pid} acquired resource: {self.resource}")
        self.locked = True
        return self.resource_file

      except IOError as io_error:
        if io_error.errno == errno.EAGAIN:
          self.logger.info(f"Process: {self.pid} attempted and failed to acquire resource: {self.resource}")
          sleep_time = 120 + retries * 60 + random.uniform(0, 60)
          time.sleep(sleep_time)
          retries += 1
        else:
          raise

  def __exit__(self, exc_type, exc_value, exc_traceback):
    if self.resource_file and self.locked:
      fcntl.flock(self.resource_file, fcntl.LOCK_UN)
      self.resource_file.close()
      self.locked = False


class IOObject():
  def __init__(self, path: str | Path) -> None:
    if isinstance(path, str):
      self._path = Path(path)
    else:
      self._path = path

  @property
  def path(self):
    return self._path

  def __eq__(self, value: object) -> bool:
    if not isinstance(value, IOObject):
      raise TypeError("Cannot compare IOObject to non objects")

    return self.path == value.path


class File(IOObject):
  pass

class Directory(IOObject):
  def __init__(self, path: str | Path, create_if_missing=True) -> None:
    super().__init__(path)

    if self.path.exists():
      if not self.path.is_dir():
        raise NotADirectoryError(f"Path {self.path} is not a directory!")

    elif create_if_missing:
      self.path.mkdir(parents=True, exist_ok=True)
    else:
      raise FileNotFoundError(f"Path {self.path} should exists as directory but it doesn't!")

  def __truediv__(self, arg) -> Path:
    return self.path / arg

  def __iter__(self):
    return self.path.iterdir()


class FileType(Enum):
  FILE = 1
  DIRECTORY = 2

def verify_path(path: Path | str, file_type: FileType) -> bool:
  """Checks if a path is as the expected file type

  Args:
    path: (Path | str) - The path to check
    file_type: (FileType) - The expected content of the path

  Returns:
    boolean: True if the path holds expected IO object

  """
  if isinstance(path, str):
    path = Path(path)

  if not path.exists():
    return False

  match file_type:
    case FileType.FILE:
      if not path.is_file():
        return False

    case FileType.DIRECTORY:
      if not path.is_dir():
        return False

  return True


TIOObject = TypeVar('TIOObject', bound=IOObject)

def parse_path(path: str | Path, file_type: Type[TIOObject]) -> TIOObject:
  return file_type(path)