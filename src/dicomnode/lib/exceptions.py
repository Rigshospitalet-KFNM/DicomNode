class IncorrectlyConfigured(Exception):
  """Server attempts to start incorrectly configured"""

class CouldNotCompleteDIMSEMessage(Exception):
  """A DIMSE message was unsuccessful"""

class InvalidRootDataDirectory(Exception):
  """creating a pipelineTree with invalid root data directory"""

class BlueprintConstructionFailure(Exception):
  """Blueprint contains a tag reserved by Dicomnode"""

class HeaderConstructionFailure(Exception):
  """When a DicomFactory fails to construct a header"""

class InvalidDataset(Exception):
  """Adding a dataset, which doesn't fulfil some requirements"""

class InvalidTagType(Exception):
  """Added an invalid type of tag"""

class InvalidQueryDataset(InvalidDataset):
  """When sending DIMSE message Querying with an invalid dataset"""

class MissingModule(ImportError):
  """Missing an module"""

class MissingPivotDataset(ValueError):
  """When iterating over an dataset iterator and by assumption it has a length
  greater than 1. This exception is raised when this assumption is broken"""

class InvalidEncoding(Exception):
  """When a data structure have an unexpected encoding"""

class InvalidTreeNode(Exception):
  """If there's an incorrect datatype in an ImageTree"""

class InvalidLatexCompiler(Exception):
  """If a compiler that should be available is unavailable"""

class InvalidFont(Exception):
  """When a font isn't a OTF or a TTF font"""

class InvalidPynetdicomEvent(Exception):
  """When a pynetdicom event is belongs to an unexcepted type"""

class MissingNiftiImage(Exception):
  """When a nibabel Nifti image doesn't contain an image"""

class MissingOptionalDependency(ImportError):
  """Raised when trying to import an optional dependency, that's missing"""