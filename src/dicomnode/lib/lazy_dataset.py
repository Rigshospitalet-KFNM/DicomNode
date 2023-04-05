"""This is an extention of the pydicom dataset. It creates a lazy dataset,
ie a Dataset that have a very small memory print until you actually use it.

Must code have been shamelessly stolen from https://coderbook.com/python/2020/04/23/how-to-make-lazy-python.html
"""

__author__ = "Christoffer Vilstrup Jensen"

# Python Standard Library
from pathlib import Path
import operator
from typing import Callable

# Thrid Party Operator
from pydicom import Dataset

# Dicomnode packages
from dicomnode.lib.io import load_dicom

class LazyDataset(Dataset): # It's not need to set this as a dataset, since we overwrite it later, however typechecker can't figure out my magic
  _wrapped = None
  _is_init = False

  def __init__(self, path):
    # Assign using __dict__ to avoid the setattr method.
    self.__dict__['_path'] = path

  def _setup(self):
    self._wrapped = load_dicom(self._path)
    self._is_init = True

  def new_method_proxy(func): # type: ignore # Dont call this method...
    """
      Util function to help us route functions
      to the nested object.
    """
    def inner(self, *args, **kwargs):
      if not self._is_init:
        self._setup()
      return func(self._wrapped, *args, **kwargs) #type: ignore
    return inner

  def __setattr__(self, name, value):
    # These are special names that are on the LazyObject.
    # every other attribute should be on the wrapped object.
    if name in {"_is_init", "_wrapped"}:
      self.__dict__[name] = value
    else:
      if not self._is_init:
        self._setup()
      setattr(self._wrapped, name, value)

  def __delattr__(self, name):
    if name == "_wrapped":
      raise TypeError("can't delete _wrapped.")
    if not self._is_init:
      self._setup()
    delattr(self._wrapped, name)

  __getattr__ = new_method_proxy(getattr) # type: ignore
  __bytes__ = new_method_proxy(bytes) # type: ignore
  __str__ = new_method_proxy(str) # type: ignore
  __bool__ = new_method_proxy(bool) # type: ignore
  __dir__ = new_method_proxy(dir) # type: ignore
  __hash__ = new_method_proxy(hash) # type: ignore
  __class__ = property(new_method_proxy(operator.attrgetter("__class__"))) # type: ignore
  __eq__ = new_method_proxy(operator.eq) # type: ignore
  __lt__ = new_method_proxy(operator.lt) # type: ignore
  __gt__ = new_method_proxy(operator.gt) # type: ignore
  __ne__ = new_method_proxy(operator.ne) # type: ignore
  __hash__ = new_method_proxy(hash) # type: ignore
  __getitem__ = new_method_proxy(operator.getitem) # type: ignore
  __setitem__ = new_method_proxy(operator.setitem) # type: ignore
  __delitem__ = new_method_proxy(operator.delitem) # type: ignore
  __iter__ = new_method_proxy(iter) # type: ignore
  __len__ = new_method_proxy(len) # type: ignore
  __contains__ = new_method_proxy(operator.contains) # type: ignore
