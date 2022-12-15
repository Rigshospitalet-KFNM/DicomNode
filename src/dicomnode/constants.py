from pydicom.uid import UID

# These are the names of the environment variables not the values of them
DICOMNODE_ENV_LOG_PATH  = "DICOMNODE_ENV_LOG_PATH"
DICOMNODE_ENV_DATA_PATH = "DICOMNODE_ENV_DATA_PATH"
DICOMNODE_ENV_FONT_PATH = "DICOMNODE_ENV_FONT_PATH"


# UID graciously provided by Medical Connections
# At https://www.medicalconnections.co.uk/FreeUID.html
DICOMNODE_IMPLEMENTATION_UID = UID('1.2.826.0.1.3680043.10.1083')

DICOMNODE_IMPLEMENTATION_NAME = "DICOMNODE"
DICOMNODE_VERSION = '0.0.2'

DICOMNODE_PRIVATE_TAG_GROUP = 0x1337 # I might regret this choice.

