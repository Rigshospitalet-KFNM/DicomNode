"""Init module mostly here to create a import tree. 
and get the version
"""

from importlib import import_module
from os import environ
from pathlib import Path


from . import constants

__version__ = constants.DICOMNODE_VERSION # This is gonna be annoying
__author__ = "Christoffer Vilstrup Jensen"

def version() -> str:
  """Gets the version of the package

  Returns:
      str: a string containing the version number separated by dots
  """
  return __version__


class _LibraryPaths:
  """Class for holding various path used by the library.

  These are intended to be static for the lifetime of the program.
  However 
  """


  # These are uninitialzed values
  _working_directory = Path('.')
  _report_directory = Path('.')
  _report_data_directory = Path('.')
  _figure_directory = Path('.')


  @property
  def working_directory(self) -> Path:
    return self._working_directory

  @property
  def report_directory(self) -> Path:
    return self._report_directory

  @property
  def report_data_directory(self) -> Path:
    return self._report_data_directory

  @property
  def figure_directory(self):
    return self._figure_directory

  def __init__(self) -> None:
    self.update_paths()

  def update_paths(self):
    """_summary_
    """
    self.set_directory_path('_processing_directory', constants.DICOMNODE_ENV_PROCESSING_PATH, constants.DEFAULT_PROCESSING_DIRECTORY)
    self.set_directory_path('_report_directory', constants.DICOMNODE_ENV_REPORT_PATH, constants.DEFAULT_REPORTS_DIRECTORY)
    self.set_directory_path('_report_data_directory', constants.DICOMNODE_ENV_REPORT_DATA_PATH, constants.DEFAULT_REPORT_DATA_DIRECTORY)
    self.set_directory_path('_figure_directory', constants.DICOMNODE_ENV_FIGURE_PATH, constants.DEFAULT_FIGURE_DIRECTORY)

  def set_directory_path(self, key: str, environment_key, default):
    if environment_key in environ:
      path = Path(environ[environment_key])
      setattr(self, key, path)
    else:
      path = Path(default)
      setattr(self, key, path)

    if not path.exists():
      path.mkdir(parents=True, exist_ok=True) # Mainly here in cases of multithreading

library_paths = _LibraryPaths()

__lib = None
__tools = None
__server = None
__report = None

# See https://peps.python.org/pep-0562/ to understand the Voodoo magic going on here
def __getattr__(name):
  if name == 'lib':
    global __lib
    if __lib is None:
      __lib = import_module('dicomnode.lib')
    return __lib
  if name == 'tools':
    global __tools
    if __tools is None:
      __tools = import_module('dicomnode.tools')
    return __tools
  if name == 'server':
    global __server
    if __server is None:
      __server = import_module('dicomnode.server')
    return __server
  if name == 'report':
    global __report
    if __report is None:
      __report = import_module('dicomnode.report')
    return __report
  raise AttributeError(f"module {__name__} has no attribute '{name}'")

__all__ = (
  'constants',
  'lib',
  'report',
  'server',
  'tools',
  'version',
  'library_paths'
)

def __dir__():
  return __all__




