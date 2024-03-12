"""This module contains various constants used throughout the module,

"""
from pydicom.uid import UID

DICOMNODE_LOGGER_NAME = 'dicomnode'

# Default paths
DEFAULT_PROCESSING_DIRECTORY = "/tmp/dicomnode"
"""Default for the environment variable: DICOMNODE_ENV_PROCESSING_PATH"""
DEFAULT_FIGURE_DIRECTORY = "/tmp/dicomnode/figures"
"""Default for the environment variable: DICOMNODE_ENV_FIGURE_PATH"""
DEFAULT_REPORTS_DIRECTORY = "/tmp/dicomnode/reports"
"""Default for the environment variable: DICOMNODE_ENV_REPORT_PATH"""
DEFAULT_REPORT_DATA_DIRECTORY = "/tmp/dicomnode/report_data"
"""Default for the environment variable: DICOMNODE_ENV_REPORT_DATA_PATH"""

# These are the names of the environment variables not the values of them
DICOMNODE_ENV_LOG_PATH  = "DICOMNODE_ENV_LOG_PATH"
DICOMNODE_ENV_REPORT_DATA_PATH = "DICOMNODE_ENV_REPORT_DATA_PATH"
DICOMNODE_ENV_FONT = "DICOMNODE_ENV_FONT"
DICOMNODE_ENV_FIGURE_PATH = "DICOMNODE_ENV_FIGURE_PATH"
DICOMNODE_ENV_PROCESSING_PATH = "DICOMNODE_ENV_PROCESSING_PATH"
DICOMNODE_ENV_REPORT_PATH = "DICOMNODE_ENV_REPORT_PATH"





# UID graciously provided by Medical Connections
# At https://www.medicalconnections.co.uk/FreeUID.html
DICOMNODE_IMPLEMENTATION_UID = UID('1.2.826.0.1.3680043.10.1083')
"""UID of this software library """


# Remember this these need to less than 16 characters!
# Be cause they are stored in a SH
DICOMNODE_IMPLEMENTATION_NAME = "DICOMNODE"
"""Name of software, to be placed in Manufacturer Models name (0008,1090)"""

DICOMNODE_VERSION = "0.0.4"
"""Version of the library"""

