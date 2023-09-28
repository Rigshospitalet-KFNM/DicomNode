"""Init module mostly here to create a import tree. 
and get the version

"""
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
