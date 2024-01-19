"""Init module mostly here to create a import tree. 
and get the version

"""
from os import environ
from pathlib import Path

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

  @property
  def working_directory(self) -> Path:
    return self._working_directory
  
  @property
  def report_directory(self) -> Path:
    return self._report_directory
  
  @property
  def report_data_directory(self) -> Path:
    return self._report_data_directory

  def __init__(self) -> None:
    self.update_paths()

  def update_paths(self):
    self.__set_directory_path('_working_directory', constants.DICOMNODE_ENV_WORKING_PATH, constants.DEFAULT_WORKING_DIRECTORY)
    self.__set_directory_path('_report_directory', constants.DICOMNODE_ENV_REPORT_PATH, constants.DEFAULT_REPORTS_DIRECTORY)
    self.__set_directory_path('_report_data_directory', constants.DICOMNODE_ENV_REPORT_DATA_PATH, constants.DEFAULT_REPORT_DATA_DIRECTORY)

  def __set_directory_path(self, key: str, environment_key, default):
    if environment_key in environ:
      path = Path(environ[environment_key])
      setattr(self, key, path)
    else:
      path = Path(default)
      setattr(self, key, path)

    if not path.exists():
      path.mkdir(parents=True, exist_ok=True) # Mainly here in cases of multithreading

library_paths = _LibraryPaths()