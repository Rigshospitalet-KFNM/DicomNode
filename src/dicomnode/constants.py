"""This module contains various constants used throughout the module,

"""
from pydicom.uid import UID

DICOMNODE_LOGGER_NAME = 'dicomnode'


# These are the names of the environment variables not the values of them
DICOMNODE_ENV_LOG_PATH  = "DICOMNODE_ENV_LOG_PATH"
DICOMNODE_ENV_DATA_PATH = "DICOMNODE_ENV_DATA_PATH"
DICOMNODE_ENV_FONT_PATH = "DICOMNODE_ENV_FONT_PATH"


# UID graciously provided by Medical Connections
# At https://www.medicalconnections.co.uk/FreeUID.html
DICOMNODE_IMPLEMENTATION_UID = UID('1.2.826.0.1.3680043.10.1083')


# Remember this these need to less than 16 characters!
# Be cause they are stored in a SH
DICOMNODE_IMPLEMENTATION_NAME = "DICOMNODE"
DICOMNODE_VERSION = "0.0.4"


