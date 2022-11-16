from pathlib import Path
from pprint import pprint, pformat
from typing import Any, List, Optional, Union, Dict, Callable

from pydicom import Dataset, FileDataset, write_file
from pydicom.uid import UID, generate_uid
from math import ceil, log10

from abc import ABC, abstractclassmethod, abstractmethod

from dicomnode.lib.utils import prefixInt

_PPrefix = "AnonymizedPatientID_"

class IdentityMapping():
  """

  Programmer Note: This class is here instead of lib.anonymization to
  prevent circular imports, for typings sake.
    Note to the Note: It might be possible to resolve it with a type hint
    'dicomnode.lib.studyTree.DicomTree'
  """
  def __init__(self, prefixSize = 4) -> None:
    self.StudyUIDMapping : Dict[str, UID] = {}
    self.SeriesUIDMapping : Dict[str, UID] = {}
    self.SOPUIDMapping : Dict[str, UID] = {}
    self.PatientMapping : Dict[str, str] = {}
    self.prefixSize = prefixSize

  def _add_to_mapping(self, uid : str , mapping : Dict) -> UID:
    if uid in mapping:
      return mapping[uid]
    else:
      mapping[uid] = generate_uid() # Well Here we include some clever prefix
      return mapping[uid]

  def add_StudyUID(self, StudyInstanceUID : UID) -> UID:
    return self._add_to_mapping(StudyInstanceUID.name, self.StudyUIDMapping)

  def add_SeriesUID(self, SeriesInstanceUID : UID) -> UID :
    return self._add_to_mapping(SeriesInstanceUID.name, self.SeriesUIDMapping)

  def add_SOPUID(self, SOPInstanceUID : UID) -> UID:
    return self._add_to_mapping(SOPInstanceUID.name, self.SOPUIDMapping)

  def add_Patient(self, PatientID : str, patient_prefix : str = _PPrefix  ) -> str:
    if PatientID in self.PatientMapping:
      return self.PatientMapping[PatientID]
    else:
      anonymized_PatientID = f"{patient_prefix}{prefixInt(len(self.PatientMapping), self.prefixSize)}"
      self.PatientMapping[PatientID] = anonymized_PatientID
      return anonymized_PatientID


  def fill_from_SeriesTree(self, seriesTree: 'SeriesTree'):
    for SOPInstanceUID, _dataSet in seriesTree.data.items():
      self._add_to_mapping(SOPInstanceUID, self.SOPUIDMapping)

  def fill_from_StudyTree(self, studyTree : 'StudyTree'):
    for seriesInstanceUID, seriesTree in studyTree.data.items():
      self._add_to_mapping(seriesInstanceUID, self.SeriesUIDMapping)
      self.fill_from_SeriesTree(seriesTree)

  def fill_from_PatientTree(self, patientTree : 'PatientTree'):
    for studyInstanceUID, studyTree in patientTree.data.items():
      self._add_to_mapping(studyInstanceUID, self.StudyUIDMapping)
      self.fill_from_StudyTree(studyTree)

  def fill_from_DicomTree(self, dicomTree : 'DicomTree', patient_prefix : str = _PPrefix, change_UIDs : bool = True):
    self.prefixSize = max(ceil(log10(len(dicomTree.data))),1)
    for patientID, studyTree in dicomTree.data.items():
      self.add_Patient(patientID, patient_prefix)
      if change_UIDs:
        self.fill_from_PatientTree(studyTree)


  def get_mapping(self, uid : Union[UID, str]) -> Optional[Union[UID, str]]:
    if isinstance(uid, UID):
      uid = uid.name
      if uid in self.StudyUIDMapping:
        return self.StudyUIDMapping[uid]
      if uid in self.SeriesUIDMapping:
        return self.SeriesUIDMapping[uid]
      if uid in self.SOPUIDMapping:
        return self.SOPUIDMapping[uid]
    else:
      if uid in self.PatientMapping:
        return self.PatientMapping[uid]
      if uid in self.StudyUIDMapping:
        return self.StudyUIDMapping[uid]
      if uid in self.SeriesUIDMapping:
        return self.SeriesUIDMapping[uid]
      if uid in self.SOPUIDMapping:
        return self.SOPUIDMapping[uid]
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
    if len(self.SOPUIDMapping) > 0:
      base_string += f"\n  SOP Mapping with {len(self.SOPUIDMapping)} Mappings"

    return base_string

class ImageTreeInterface(ABC):
  @abstractmethod
  def add_image(self, _dicom : Dataset) -> None:
    raise NotImplemented #pragma: no cover

  def add_images(self, listOfDicom : List[Dataset]) -> None:
    for dicom in listOfDicom:
      self.add_image(dicom)

  def _map(self, func : Callable[[Dataset], Any],
                    index_map : Optional[Dict],
                    UIDMapping : Optional[IdentityMapping] = None) -> Dict[str, Any]:
    new_data = {}
    ret_dir = {}
    for ID, tree in self.data.items():
      ret_dir.update(tree.map(func, UIDMapping))
      if index_map:
        if ID in index_map:
          new_data[index_map[ID]] = tree
        else:
          new_data[ID] = tree
      else:
        new_data[ID] = tree
    self.data = new_data
    return ret_dir

  @abstractmethod
  def map(self, func: Callable[[Dataset], Any],
                    UIDMapping: Optional[IdentityMapping] = None) -> Dict[str, Any]:
    """Applies a callable function to all dataset in the Tree.
    If the function changes the keys, namly:
      SOPInstanceUID
      SeriesInstanceUID
      StudyInstanceUID
      PatientID
    Then that should be included as the UIDMapping

    Args:
        func (Callable[[Dataset], ]): Function to be applied to each dataset
        UIDMapping (Optional[IdentityMapping], optional): Mapping of UID to applied to the TreeInterface. Defaults to None.
    """
    raise NotImplemented #pragma: no cover

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
        filterfunc (Callable[[Dataset], bool]): _description_

    Returns:
        int: Number of Pictures trimmed
    """
    trimmed_total:int = 0
    new_data: Dict[str, Union[Dataset, ImageTreeInterface]] = {}
    for ID, tree in self.data.items():
      trimmed = tree.trim_tree(filter_function)
      if tree.images == 0:
        pass # Delete the entry
      elif tree.images < 0: #pragma: no cover
        raise ValueError("Removed more images than possible!") #pragma: no cover
      else: # tree.images > 0
        new_data[ID] = tree
      trimmed_total += trimmed
      self.images -= trimmed
    self.data = new_data
    return trimmed_total

  def __init__(self, dcm: Optional[Union[List[Dataset], Dataset]] = None) -> None:
    self.data: Dict[str, Union[Dataset, ImageTreeInterface]] = {} # Dict containing Images, Series, Studies or Patients
    self.images: int = 0 # The total number of images of this tree and it's subtrees

    if dcm:
      if isinstance(dcm, Dataset):
        self.add_image(dcm)
      else:
        self.add_images(dcm)


class SeriesTree(ImageTreeInterface):
  """Final Layer of the DicomTree that contains the data.
  """
  SeriesDescription = "Tree of Undefined Series"

  def add_image(self, dicom : Dataset) -> None:
    if not hasattr(dicom, 'SeriesInstanceUID'):
      raise ValueError("Dicom image doesn't have a SeriesInstanceUID")
    if not hasattr(dicom, 'SOPInstanceUID'):
      raise ValueError("Dicom image doesn't have a SOPInstanceUID")
    if hasattr(self, 'SeriesInstanceID'):
      if self.SeriesInstanceUID != dicom.SeriesInstanceUID.name:
        raise KeyError("Attempting to add an image to a series where it doesn't belog")
    else:
      self.SeriesInstanceUID = dicom.SeriesInstanceUID.name
      if hasattr(dicom, 'SeriesDescription'):
        self.SeriesDescription = f"Tree of {dicom.SeriesDescription}"
    if dicom.SOPInstanceUID.name in self.data:
      raise ValueError("Dublicate Image added!")
    self.data[dicom.SOPInstanceUID.name] = dicom
    self.images += 1

  def map(self, func: Callable[[Dataset], Any], UIDMapping: Optional[IdentityMapping] = None) -> Dict[str, Any]:
    new_data = {}
    ret_dict = {}
    for SOPInstanceUID, dataset in self.data.items():
      ret_val = func(dataset)
      if UIDMapping:
        if SOPInstanceUID in UIDMapping.SOPUIDMapping:
          newSOPinstance = UIDMapping.SOPUIDMapping[SOPInstanceUID]
          new_data[newSOPinstance] = dataset
          ret_dict[newSOPinstance] = ret_val
        else:
          new_data[newSOPinstance] = dataset
          ret_dict[newSOPinstance] = ret_val
      else:
        new_data[SOPInstanceUID] = dataset
        ret_dict[SOPInstanceUID] = ret_val
    self.data = new_data
    return ret_dict

  def trim_tree(self, filterfunc: Callable[[Dataset], bool]) -> int:
    trimmed = 0
    new_data = {}
    for SOPInstanceUID, dataset in self.data.items():
      if filterfunc(dataset):
        new_data[SOPInstanceUID] = dataset
      else:
        trimmed += 1
        self.images -= 1
    self.data = new_data
    return trimmed

  def save_tree(self, target: Path) -> None:
    if len(self.data) == 1:
      for _, v in self.data.items(): # Only Iterated once
        write_file(target, v)
    else:
      target.mkdir()
      for k, v in self.data.items(): # Only Iterated once
        file_target = target / k
        write_file(file_target, v)

  def __str__(self) -> str:
    return f"{self.SeriesDescription} with {self.images} images"

class StudyTree(ImageTreeInterface):
  """A Study tree is a data object that contains all studies with the same study ID
  """
  StudyDescription = f"Undefined Study Description"

  def add_image(self, dicom : Dataset) -> None:
    if not hasattr(dicom, 'StudyInstanceUID'):
      raise ValueError("Dicom image doesn't have a StudyInstanceUID")
    if not hasattr(dicom, 'SeriesInstanceUID'):
      raise ValueError("Dicom image doesn't have a SeriesInstanceUID")
    if hasattr(self, 'StudyInstanceID'):
      if self.StudyInstanceUID != dicom.StudyInstanceUID.name:
        raise KeyError("Attempting to add an image to a study where it doesn't belog")
    else:
      self.StudyInstanceUID = dicom.StudyInstanceUID.name
      if hasattr(dicom, 'StudyDescription'):
        self.StudyDescription = f"Tree of {dicom.StudyDescription}"
    if dicom.SeriesInstanceUID.name in self.data:
      tree = self.data[dicom.SeriesInstanceUID.name]
      tree.add_image(dicom)
    else:
      self.data[dicom.SeriesInstanceUID.name] = SeriesTree(dicom)
    self.images += 1

  def map(self, func: Callable[[Dataset], Any], UIDMapping: Optional[IdentityMapping] = None) -> Dict[str, Any]:
    if UIDMapping:
      return self._map(func, UIDMapping.SeriesUIDMapping, UIDMapping)
    else:
      return self._map(func, None, None)

  def __str__(self) -> str:
    seriesStr = f""
    for series in self.data.values():
      seriesStr += f"      {series}\n"
    return f"{self.StudyDescription} with {self.images} images with Series:\n{seriesStr}"


class PatientTree(ImageTreeInterface):
  """A Tree of Dicom images under one patient, based around Patient ID
  """
  TreeName : str = "Unknown Tree"

  def add_image(self, dicom : Dataset) -> None:
    if not hasattr(dicom, 'StudyInstanceUID'):
      raise ValueError("Dicom image doesn't have StudyInstanceUID")
    if not hasattr(dicom, 'PatientID'):
      raise ValueError("Dicom image doesn't have PatientID")
    if hasattr(self, 'PatientID'):
      if self.PatientID != dicom.PatientID:
        raise KeyError("Attempting to add an image of an patient to the wrong tree")
    else:
      self.PaitentID = dicom.PatientID
      if hasattr(dicom, 'PatientName'):
        self.TreeName = f"StudyTree of {dicom.PatientName}"
    if tree := self.data.get(dicom.StudyInstanceUID.name):
      tree.add_image(dicom)
    else:
      self.data[dicom.StudyInstanceUID.name] = StudyTree(dicom)
    self.images += 1

  def map(self, func: Callable[[Dataset], Any], UIDMapping: Optional[IdentityMapping] = None) -> Dict[str, Any]:
    if UIDMapping:
      return self._map(func, UIDMapping.StudyUIDMapping, UIDMapping)
    else:
      return self._map(func, None, None)


  def __str__(self) -> str:
    studyStr = ""
    for study in self.data.values():
      studyStr += f"    {study}"
    return f"Patient {self.TreeName} with {self.images} images\n{studyStr}"

class DicomTree(ImageTreeInterface):
  """This is a Root node of an ImageTree structure that sort Dicom Images in the following way:

  The Structure is as follows:
                     DicomTree\n
                    /   ...   \ 
            PatientTree...\n
            /   ...    \ 
          StudyTree...\n
          /    ...  \ 
     SeriesTree...\n
    /   ...   \ 
  DataSet...
  """

  def add_image(self, dicom : Dataset) -> None:
    if not hasattr(dicom, 'PatientID'):
      raise ValueError("Dicom Image doesn't have PatientID")
    if tree := self.data.get(dicom.PatientID):
      tree.add_image(dicom)
    else:
      tree = PatientTree()
      tree.add_image(dicom)
      self.data[dicom.PatientID] = tree
    self.images += 1

  def map(self, func: Callable[[Dataset], Any], UIDMapping: Optional[IdentityMapping] = None) -> Dict[str, Any]:
    if UIDMapping:
      return self._map(func, UIDMapping.PatientMapping, UIDMapping)
    else:
      return self._map(func, None, None)

  def __str__(self) -> str:
    patientStr = f""
    for patientTree in self.data.values():
      patientStr += f"  {patientTree}"
    return f"Dicom Tree with {self.images} images\n{patientStr}"

