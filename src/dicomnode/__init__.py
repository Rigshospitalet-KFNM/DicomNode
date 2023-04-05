__version__ = "0.0.3" # This is gonna be annoying

__author__ = "Christoffer Vilstrup Jensen"

from . import lib
from . import tools
from . import server
from . import constants
from . import report

def version() -> str:
  return __version__
