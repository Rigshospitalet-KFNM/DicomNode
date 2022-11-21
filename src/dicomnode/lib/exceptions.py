class IncorrectlyConfigured(Exception):
  """Raised when the server attempt to start, but have not been correctly
  configured."""

class CouldNotCompleteDIMSEMessage(Exception):
  """Raised when a user attempts to send a DIMSE message,
  when it's unsuccessful this message is raised
  """

class InvalidRootDataDirectory(Exception):
  """Raised when attempting to create a pipelineTree and
  the root data directory is invalid
  """

class InvalidDataset(Exception):
  """Raised when attempting to add dataset, which doesn't fulfil some requirements"""

class InvalidQueryDataset(InvalidDataset):
  """Raised when attempting to query with an invalid dataset"""