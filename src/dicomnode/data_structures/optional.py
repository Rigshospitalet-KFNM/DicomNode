
from pathlib import Path
from typing import Any, Callable, Optional

from dicomnode.lib.exceptions import ContractViolation

class OptionalPath:
  def __init__(self, path: Optional[Any]= None) -> None:
    if path is not None and not isinstance(path, Path):
      path = Path(path)
    self.value = path

  def __truediv__(self, other):
    if self.value is None:
      return OptionalPath()

    return OptionalPath(self.value / other)

  def __bool__(self):
    return self.value is not None

  def __call__(self, callable_: Callable[[Path], Any]) -> Any:
    if self.value is not None:
      return callable_(self.path)
    return None

  @property
  def path(self) -> Path:
    if self.value is None:
      raise ContractViolation()
    return self.value
