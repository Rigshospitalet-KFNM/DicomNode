class IncorrectlyConfigured(Exception):
  """Raised when the server attempt to start, but have not been correctly configured."""

  pass

class CouldNotCompleteDIMSEMessage(Exception):
  """Raised when a user attempts to send a DIMSE message, when it's unsuccessful this message is raised"""