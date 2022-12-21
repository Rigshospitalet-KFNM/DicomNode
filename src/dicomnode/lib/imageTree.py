import logging
from abc import ABC, abstractclassmethod, abstractmethod
from math import ceil, log10
from pathlib import Path
from pprint import pformat, pprint
from typing import (Any, Callable, Dict, Iterable, Iterator, List, Optional,
                    Union)

from psutil import virtual_memory
from pydicom import Dataset, FileDataset, write_file
from pydicom.errors import InvalidDicomError
from pydicom.uid import UID

from dicomnode.lib.dicom import gen_uid
from dicomnode.lib.exceptions import InvalidTreeNode
from dicomnode.lib.io import load_dicom, save_dicom
from dicomnode.lib.utils import prefixInt

_PPrefix = "AnonymizedPatientID_"

logger = logging.getLogger("dicomnode")

class IdentityMapping():
  """

  Programmer Note: This class is here instead of lib.anonymization to
  prevent circular imports, for typings sake.
    Note to the Note: It might be possible to resolve it with a type hint
    'dicomnode.lib.studyTree.DicomTree'
  """
  def __init__(self, prefix_size = 4) -> None:
    self.StudyUIDMapping : Dict[str, UID] = {}
    self.SeriesUIDMapping : Dict[str, UID] = {}
    self.SOP_UIDMapping : Dict[str, UID] = {}
    self.PatientMapping : Dict[str, str] = {}
    self.prefix_size = prefix_size

  def __contains__(self, key: str) -> bool:
    return key in self.StudyUIDMapping \
            or key in self.SeriesUIDMapping \
            or key in self.SOP_UIDMapping \
            or key in self.PatientMapping

  def __add_to_mapping(self, uid : str , mapping : Dict) -> UID:
    if uid in mapping:
      return mapping[uid]
    else:
      mapping[uid] = gen_uid() # Well Here we include some clever prefix
      return mapping[uid]

  def add_StudyUID(self, StudyInstanceUID : UID) -> UID:
    return self.__add_to_mapping(StudyInstanceUID.name, self.StudyUIDMapping)

  def add_SeriesUID(self, SeriesInstanceUID : UID) -> UID :
    return self.__add_to_mapping(SeriesInstanceUID.name, self.SeriesUIDMapping)

  def add_SOP_UID(self, SOPInstanceUID : UID) -> UID:
    return self.__add_to_mapping(SOPInstanceUID.name, self.SOP_UIDMapping)

  def add_Patient(self, PatientID : str, patient_prefix : str = _PPrefix  ) -> str:
    if PatientID in self.PatientMapping:
      return self.PatientMapping[PatientID]
    else:
      anonymized_PatientID = f"{patient_prefix}{prefixInt(len(self.PatientMapping), self.prefix_size)}"
      self.PatientMapping[PatientID] = anonymized_PatientID
      return anonymized_PatientID


  def fill_from_SeriesTree(self, seriesTree: 'SeriesTree'):
    for SOPInstanceUID, _dataSet in seriesTree.data.items():
      self.__add_to_mapping(SOPInstanceUID, self.SOP_UIDMapping)

  def fill_from_StudyTree(self, studyTree : 'StudyTree'):
    for seriesInstanceUID, seriesTree in studyTree.data.items():
      self.__add_to_mapping(seriesInstanceUID, self.SeriesUIDMapping)
      if isinstance(seriesTree, SeriesTree):
        self.fill_from_SeriesTree(seriesTree)
      else:
        raise InvalidTreeNode # pragma: no cover

  def fill_from_PatientTree(self, patientTree : 'PatientTree'):
    for studyInstanceUID, studyTree in patientTree.data.items():
      self.__add_to_mapping(studyInstanceUID, self.StudyUIDMapping)
      if isinstance(studyTree, StudyTree):
        self.fill_from_StudyTree(studyTree)
      else:
        raise InvalidTreeNode # pragma: no cover

  def fill_from_DicomTree(self, dicomTree : 'DicomTree', patient_prefix : str = _PPrefix, change_UIDs : bool = True):
    self.prefix_size = max(ceil(log10(len(dicomTree.data))),1)
    for patientID, patientTree in dicomTree.data.items():
      self.add_Patient(patientID, patient_prefix)

      if change_UIDs:
        if isinstance(patientTree, PatientTree):
          self.fill_from_PatientTree(patientTree)
        else:
          raise InvalidTreeNode # pragma: no cover

  def __getitem__(self, key: Union[UID, str]):
    if isinstance(key, UID):
      key = key.name

    if key in self.PatientMapping:
      return self.PatientMapping[key]
    if key in self.StudyUIDMapping:
      return self.StudyUIDMapping[key]
    if key in self.SeriesUIDMapping:
      return self.SeriesUIDMapping[key]
    if key in self.SOP_UIDMapping:
      return self.SOP_UIDMapping[key]
    raise KeyError()

  def get_mapping(self, uid : Union[UID, str]) -> Optional[Union[UID, str]]:
    if isinstance(uid, UID):
      uid = uid.name

    if uid in self.PatientMapping:
      return self.PatientMapping[uid]
    if uid in self.StudyUIDMapping:
      return self.StudyUIDMapping[uid]
    if uid in self.SeriesUIDMapping:
      return self.SeriesUIDMapping[uid]
    if uid in self.SOP_UIDMapping:
      return self.SOP_UIDMapping[uid]
    return None

  def __str__(self) -> str:
    base_string = f"Identity Mapping"
    # Patients
    if len(self.PatientMapping) > 0:
      base_string += f"\n  Patient Mapping\n{pformat(self.PatientMapping, indent=4)}"
    # Studies
    if len(self.StudyUIDMapping) > 0:
      base_string += f"\n  Study Mapping with {len(self.StudyUIDMapping)} Mappings"
        # Series
    if len(self.SeriesUIDMapping) > 0:
      base_string += f"\n  Series Mapping with {len(self.SeriesUIDMapping)}"
    # SOP instances
    if len(self.SOP_UIDMapping) > 0:
      base_string += f"\n  SOP Mapping with {len(self.SOP_UIDMapping)} Mappings"

    return base_string

class ImageTreeInterface(ABC):
  """Base class for a tree of Dicom Images.

  """
  @abstractmethod
  def add_image(self, _dicom : Dataset) -> int:
    """Abstract Method for adding datasets to the Tree"""
    raise NotImplemented #pragma: no cover

  def add_images(self, listOfDicom : Iterable[Dataset]) -> int:
    added = 0
    for dicom in listOfDicom:
      added += self.add_image(dicom)
    return added

  def map(self,
           func : Callable[[Dataset], Any],
           UIDMapping : IdentityMapping = IdentityMapping()
    ) -> Dict[str, Any]:
    new_data: Dict[str, Union[Dataset, ImageTreeInterface]] = {}
    ret_dir = {}
    for ID, entry in self.data.items():
      if ID in UIDMapping:
        ID = UIDMapping[ID]
      if isinstance(entry, ImageTreeInterface):
        ret_dir.update(entry.map(func, UIDMapping))
        new_data[ID] = entry
      elif isinstance(entry, Dataset): # is a Dataset
        return_value = func(entry)
        if return_value is not None:
          ret_dir[ID] = return_value
        new_data[ID] = entry
      else:
        raise InvalidTreeNode("A None ImageTree or Data is added") # pragma: no cover
    self.data = new_data
    return ret_dir

  def discover(self, path: Path):
    """Fills a DicomTree with studies found at <path>.
      Recursively searches a Directory for dicom files.
      Skipping files it cannot open.

    Args:
      path (Path): Path that will be searched through
    """
    if path.is_file():
      try:
        dataset = load_dicom(path)
        mem = virtual_memory()
        if mem.available < 100*1024*1024: # This should be moved into a constants file
          print("Limited Memory available") #pragma: no cover
        self.add_image(dataset)
      except InvalidDicomError as E:
        logger.error(f"Attempting to load a none dicom file at: {path}")
    elif path.is_dir():
      for p in path.iterdir():
        self.discover(p)

  def save_tree(self, target: Path) -> None:
    if(len(self.data) == 1):
     for _, v in self.data.items(): # Only Iterated once
        v.save_tree(target)
    else:
      target.mkdir()
      for k, v in self.data.items():
        subtree_target = target / k
        v.save_tree(subtree_target)

  def trim_tree(self, filter_function : Callable[[Dataset], bool]) -> int:
    """Removes any Dataset that return false in the Filter function.
    Destroys any subtrees that zeroes images.

    Args:
        filter_function (Callable[[Dataset], bool]): _description_

    Returns:
        int: Number of Pictures trimmed
    """
    trimmed_total:int = 0
    new_data: Dict[str, Union[Dataset, ImageTreeInterface]] = {}
    for ID, tree in self.data.items():
      if isinstance(tree,  ImageTreeInterface):
        trimmed = tree.trim_tree(filter_function)
        if tree.images == 0:
          pass # Delete the entry
        elif tree.images < 0: #pragma: no cover
          raise ValueError("Removed more images than possible!") #pragma: no cover
        else: # tree.images > 0
          new_data[ID] = tree
        trimmed_total += trimmed
        self.images -= trimmed
      elif isinstance(tree, Dataset):
        if filter_function(tree):
          new_data[ID] = tree
        else:
          trimmed_total += 1
          self.images -= 1
    self.data = new_data
    return trimmed_total

  def __iter__(self) -> Iterator[Dataset]:
    for subtree in self.data.values():
      if isinstance(subtree, ImageTreeInterface):
        for subtreeVal in subtree:
          yield subtreeVal
      else:
        yield subtree

  def __len__(self):
    return self.images

  @property
  def data(self) -> Dict[str, Union[Dataset, 'ImageTreeInterface']]:
    """Data stored in the Image tree

    Leafs are datasets and Nodes are Image trees

    Throws an

    """
    return self.__data

  @data.setter
  def data(self, value: Dict[str, Union[Dataset, 'ImageTreeInterface']]):
    """This function is not callable from subclasses"""
    self.__data = value

  def __getitem__(self, key) -> Union[Dataset, 'ImageTreeInterface']:
    if isinstance(key, UID):
      key = key.name
    return self.data[key]


  def __setitem__(self, key: Union[str, UID], entry: Union[Dataset, 'ImageTreeInterface']) -> None:
    """Low Leveler function that actually stores the data.

      Use Add image for a high level version of this.
    """
    if isinstance(key, UID):
      key = key.name
    if not isinstance(key, str):
      raise TypeError("The Key should be an string")
    if not (isinstance(entry, (Dataset, ImageTreeInterface))):
      raise TypeError("The Entry should be a Dataset or An ImageTree")
    self.__data[key] = entry

  def __delitem__(self, key: Union[str, UID]) -> None:
    if isinstance(key, UID):
      key = key.name

    val = self[key]
    if isinstance(val, Dataset):
      self.images -= 1
    elif isinstance(val, ImageTreeInterface):
      self.images -= val.images
    del val # not needed but whatever
    del self.__data[key]

  def __contains__(self, key: Union[str, UID]) -> bool:
    if isinstance(key, UID):
      key = key.name
    return key in self.data

  def __init__(self, dcm: Union[Iterable[Dataset], Dataset] = []) -> None:
    self.__data: Dict[str, Union[Dataset, ImageTreeInterface]] = {} # Dict containing Images, Series, Studies or Patients
    self.images: int = 0 # The total number of images of this tree and it's subtrees


    if dcm is not None:
      if isinstance(dcm, Dataset):
        self.add_image(dcm)
      elif isinstance(dcm, Iterable):
        self.add_images(dcm)

class SeriesTree(ImageTreeInterface):
  """Final Layer of the DicomTree that contains the data.
  """

  SeriesDescription = "Tree of Undefined Series"

  def add_image(self, dicom : Dataset) -> int:
    if not hasattr(dicom, 'SeriesInstanceUID'):
      raise ValueError("Dicom image doesn't have a SeriesInstanceUID")
    if not hasattr(dicom, 'SOPInstanceUID'):
      raise ValueError("Dicom image doesn't have a SOPInstanceUID")
    if hasattr(self, 'SeriesInstanceUID'):
      if self.SeriesInstanceUID != dicom.SeriesInstanceUID.name:
        raise KeyError("Attempting to add an image to a series where it doesn't belong")
    else:
      self.SeriesInstanceUID = dicom.SeriesInstanceUID.name
      if hasattr(dicom, 'SeriesDescription'):
        self.SeriesDescription = f"Tree of {dicom.SeriesDescription}"
    if dicom.SOPInstanceUID.name in self.data:
      raise ValueError("Duplicate Image added!")
    self[dicom.SOPInstanceUID.name] = dicom
    self.images += 1
    return 1

  def save_tree(self, target: Path) -> None:
    if len(self) == 1:
      for dicom in self: # Only Iterated once
        save_dicom(target, dicom)
    else:
      target.mkdir()
      for dicom in self: # Only Iterated once
        file_target = target / dicom.SOPInstanceUID.name
        save_dicom(file_target, dicom)

  def __str__(self) -> str:
    return f"{self.SeriesDescription} with {self.images} images"

class StudyTree(ImageTreeInterface):
  """A Study tree is a data object that contains all studies with the same study ID
  """

  StudyDescription: str = f"Undefined Study Description"

  def add_image(self, dicom : Dataset) -> int:
    if not hasattr(dicom, 'StudyInstanceUID'):
      raise ValueError("Dicom image doesn't have a StudyInstanceUID")
    if not hasattr(dicom, 'SeriesInstanceUID'):
      raise ValueError("Dicom image doesn't have a SeriesInstanceUID")
    if hasattr(self, 'StudyInstanceUID'):
      if self.StudyInstanceUID != dicom.StudyInstanceUID.name:
        raise KeyError("Attempting to add an image to a study where it doesn't belong")
    else:
      self.StudyInstanceUID = dicom.StudyInstanceUID.name
      if hasattr(dicom, 'StudyDescription'):
        self.StudyDescription = f"Tree of {dicom.StudyDescription}"
    if dicom.SeriesInstanceUID.name in self.data:
      tree = self.data[dicom.SeriesInstanceUID.name]
      tree.add_image(dicom)
    else:
      self[dicom.SeriesInstanceUID.name] = SeriesTree(dicom)
    self.images += 1
    return 1

  def __str__(self) -> str:
    seriesStr = f""
    for series in self.data.values():
      seriesStr += f"      {series}\n"
    return f"{self.StudyDescription} with {self.images} images with Series:\n{seriesStr}"


class PatientTree(ImageTreeInterface):
  """A Tree of Dicom images under one patient, based around Patient ID
  """

  TreeName : str = "Unknown Tree"

  def add_image(self, dicom : Dataset) -> int:
    if not hasattr(dicom, 'StudyInstanceUID'):
      raise ValueError("Dicom image doesn't have StudyInstanceUID")
    if not hasattr(dicom, 'PatientID'):
      raise ValueError("Dicom image doesn't have PatientID")
    if hasattr(self, 'PatientID'):
      if self.PatientID != dicom.PatientID:
        raise KeyError("Attempting to add an image of an patient to the wrong tree")
    else:
      self.PatientID = dicom.PatientID
      if hasattr(dicom, 'PatientName'):
        self.TreeName = f"StudyTree of {dicom.PatientName}"
    if tree := self.data.get(dicom.StudyInstanceUID.name):
      tree.add_image(dicom)
    else:
      self[dicom.StudyInstanceUID.name] = StudyTree(dicom)
    self.images += 1
    return 1

  def __str__(self) -> str:
    studyStr = ""
    for study in self.data.values():
      studyStr += f"    {study}"
    return f"Patient {self.TreeName} with {self.images} images\n{studyStr}"

class DicomTree(ImageTreeInterface):
  """This is a Root node of an ImageTree structure that sort Dicom Images in the following way:

  The Structure is as follows:
                     DicomTree\n
                    /   ...    \\
            PatientTree...\n
            /   ...    \\
          StudyTree...\n
          /    ...  \\
     SeriesTree...\n
    /   ...   \\
  DataSet ... Dataset
  """

  def add_image(self, dicom : Dataset) -> int:
    if not hasattr(dicom, 'PatientID'):
      raise ValueError("Dicom Image doesn't have PatientID")

    if dicom.PatientID in self:
      self[dicom.PatientID].add_image(dicom)
    else:
      tree = PatientTree()
      tree.add_image(dicom)
      self[dicom.PatientID] = tree
    self.images += 1
    return 1

  def __str__(self) -> str:
    patientStr = f""
    for patientTree in self.data.values():
      patientStr += f"  {patientTree}"
    return f"Dicom Tree with {self.images} images\n{patientStr}"

