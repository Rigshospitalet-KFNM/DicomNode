from argparse import Namespace
from logging import Logger
from pathlib import Path
from typing import Dict, Optional, Tuple


from psutil import virtual_memory

import pydicom
from pydicom import Dataset, Sequence
from pydicom.values import convert_SQ, convert_string
from pydicom.datadict import DicomDictionary, keyword_dict #type: ignore Yeah Pydicom have some fancy import stuff.
from dicomnode.lib.parser import read_private_tag, PrivateTagParserReadException

def update_private_tags(new_dict_items : Dict[int, Tuple[str, str, str, str, str]]) -> None:
  """ Updated the dicom dictionary with a set of new private tags,
    allowing pydicom to reconize private tags.

  Args:
    new_dict_items : dict[int, Tuple[str, str, str, str, str]]

  Example
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
      is_little_endian (bool, optional): _description_. Defaults to True.

  Returns:
      _type_: _description_

  Exampel

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


def load_dicom(
    dicomPath: Path,
    private_tags: Optional[Dict[int, Tuple[str, str, str, str, str]]] = None
  ):
  return pydicom.dcmread(dicomPath)

def save_dicom(
    dicomPath: Path,
    dicom: Dataset
  ):
  dicomPath = dicomPath.absolute()
  if not dicomPath.parent.exists():
    dicomPath.parent.mkdir(parents=True)

  dicom.save_as(dicomPath, write_like_original=False)


def load_private_tags(dicPath : Path, strict=False) -> Dict[int, Tuple[str, str, str, str, str]]:
  private_tags = {}
  with dicPath.open() as f:
    while line := f.readline():
      try:
        parsedTags = read_private_tag(line)
        if parsedTags:
          (tag, data) = parsedTags
          private_tags[tag] = data

      except PrivateTagParserReadException as E:
        if not strict:
          raise E
        else:
          print(f"Line: {line} could not be parsed")
  return private_tags

def load_private_tags_from_args(args : Namespace) -> Dict[int, Tuple[str,str,str,str,str]]:
  """Wrapper function to load_private_tags that extracts arguments from a namespace

  Args:
      args (_type_): _description_
  """
  private_tags = {}
  if args.privatetags:
    private_tags = load_private_tags(args.privatetags, args.strictParsing)
  return private_tags
