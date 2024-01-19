"""Init module mostly here to create a import tree. 
and get the version

"""
from os import environ as __environ
from pathlib import Path as __Path

from . import lib
from . import tools
from . import server
from . import constants
from . import report

__version__ = constants.DICOMNODE_VERSION # This is gonna be annoying
__author__ = "Christoffer Vilstrup Jensen"

def version() -> str:
  """Gets the version

  Returns:
      str: _description_
  """
  return __version__


class _LibraryPaths:
  """Class for holding various path used by the library. 

  """
  working_directory = __Path("/tmp/dicomnode")

  def __init__(self) -> None:
    self.update_paths()

  def update_paths(self):
    pass

  def __set_path(self, key: str, environment_key, default):
    if environment_key in __environ:
      setattr(self, key, __Path(environment_key))
    else:
      setattr(self, key, __Path(default))

library_paths = _LibraryPaths()