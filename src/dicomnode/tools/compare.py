"""This script compares two different datasets"""

# Python standard library
from enum import Enum
from argparse import _SubParsersAction, Namespace, ArgumentParser
from pathlib import Path
from typing import Tuple, Optional

from textwrap import dedent

# Third party Packages
from pydicom.datadict import keyword_for_tag
from pydicom.dataelem import DataElement
from pydicom.dataset import Dataset

# Dicomnode packages
from dicomnode.lib import io


FIRST_KEYWORD = "fst"
SECOND_KEYWORD = "sec"

_HELP_MESSAGE = dedent("""\
  This tool compares two different datasets either by what tags or values are
  different""")

def get_print_for_tag(tag):
  return f"{tag.tag} - {keyword_for_tag(tag.tag)} - {tag.value}"

def get_print_string_1(tag: DataElement):
  return f"< {get_print_for_tag(tag)}"

def get_print_string_2(tag: DataElement):
  return f"> {get_print_for_tag(tag)}"

class SortingMethods(Enum):
  tags = "tags"
  series = "series"

def sorting_method(method: SortingMethods):
  def series(t: Tuple[DataElement, str, str]):
    return t[1]

  def tags(t: Tuple[DataElement, str, str]):
    return t[0].tag

  if method ==  SortingMethods.tags:
    return tags

  return series


class DualDicomIterator:
  def __init__(self, dicom_1, dicom_2):
    self._dicom_1 = iter(dicom_1)
    self._dicom_2 = iter(dicom_2)
    self._tag1 = next(self._dicom_1)
    self._tag2 = next(self._dicom_2)

  def iter_tag_1(self):
    try:
      self._tag1 = next(self._dicom_1)
    except StopIteration:
      self._tag1 = None

  def iter_tag_2(self):
    try:
      self._tag2 = next(self._dicom_2)
    except StopIteration:
      self._tag2 = None

  def __iter__(self):
    while True:
      try:
        yield next(self)
      except StopIteration:
        return

  def __next__(self) -> Tuple[Optional[DataElement], Optional[DataElement]]:
    while self._tag1 is not None or self._tag2 is not None:
      tag_1, tag_2 = self._tag1, self._tag2
      if tag_1 is None:
        self.iter_tag_2()
        return None, tag_2
      elif tag_2 is None:
        self.iter_tag_1()
        return tag_1, None
      elif tag_1.tag == tag_2.tag:
        self.iter_tag_1()
        self.iter_tag_2()
        return tag_1, tag_2
      elif tag_1.tag < tag_2.tag:
        self.iter_tag_1()
        return tag_1, None
      else:
        self.iter_tag_2()
        return None, tag_2
    raise StopIteration




def get_parser(subparser: _SubParsersAction):
  _, _, tool_name = __name__.split(".")

  module_parser: ArgumentParser = subparser.add_parser(tool_name, help=_HELP_MESSAGE)

  module_parser.add_argument('dicom_1', type=Path, help="The first dicom to compare")
  module_parser.add_argument('dicom_2', type=Path, help="The first dicom to compare")
  module_parser.add_argument('--compare-values',action='store_true', help="Expands the comparison to print on different values")
  module_parser.add_argument('--sorting-method',choices=SortingMethods, default=SortingMethods.tags)
  module_parser.add_argument('--only-vr', default=None)


def entry_func(args: Namespace):
  dicom_1 = io.load_dicom(args.dicom_1)
  dicom_2 = io.load_dicom(args.dicom_2)

  lines_to_print = []

  for tag_1, tag_2 in DualDicomIterator(dicom_1, dicom_2):
    if tag_1 is None and tag_2 is not None:
      lines_to_print.append((tag_2, SECOND_KEYWORD, get_print_string_2(tag_2)))
    elif tag_2 is None and tag_1 is not None:
      lines_to_print.append((tag_1, FIRST_KEYWORD,get_print_string_1(tag_1)))
    elif args.compare_values and tag_1 is not None and tag_2 is not None:
      if tag_1.value != tag_2.value and tag_1.VR not in ['OB', 'OW', 'OF']:
        lines_to_print.append((tag_1, FIRST_KEYWORD, get_print_string_1(tag_1)))
        lines_to_print.append((tag_2, SECOND_KEYWORD, get_print_string_2(tag_2)))

  sorted_lines = sorted(lines_to_print, key=sorting_method(args.sorting_method))
  if len(sorted_lines):
    print("\n".join([t[2] for t in sorted_lines]))
  else:
    print("The dataset er equal")
