from typing import cast, Callable, Dict, Optional, overload
import inspect

from dicomnode.lib.exceptions import ContractViolation


class DefaultingDict[K,V]:
  """This is a dict that default constructs an element when you try and get it.
  The main idea is that you retrieve and mutate the elements of the Dict.
  """

  @overload
  def __init__(self, callable_: Callable[[K], V]): ...
  @overload
  def __init__(self, callable_: Callable[[], V]): ...
  def __init__(self, callable_: Callable[[], V] | Callable[[K], V]) -> None:
    self._dict: Dict[K,V] = {}

    self._callable_one_arg: Optional[Callable[[K], V]] = None
    self._callable_no_args: Optional[Callable[[], V]] = None

    try:
      sig = inspect.signature(callable_)
      params = [
          p for p in sig.parameters.values()
          if p.default is inspect.Parameter.empty
          and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
      ]
      match len(params):
        case 1:
          self._callable_one_arg = cast(Callable[[K], V] ,callable_)
        case 0:
          self._callable_no_args = cast(Callable[[], V], callable_)
        case _:
          raise ContractViolation("callable must take 0 or 1 arguments")
    except (ValueError, TypeError):
      # If we can't inspect (e.g. some builtins), fall back to trying with key
      self._callable_no_args = cast(Callable[[], V], callable_)

  def __getitem__(self, key: K) -> V:
    if key not in self._dict:
      self._dict[key] = self._construct(key)

    return self._dict[key]

  def __delitem__(self, key: K) -> None:
    if key in self._dict:
      del self._dict[key]

  def __iter__(self):
    for key, value in self._dict.items():
      yield key, value

  def __len__(self):
    return len(self._dict)

  def extract(self, key: K) -> V:
    """Retrieves and removes an element

    Args:
        key (K): The key to index with

    Returns:
        V: The value at key
    """
    value = self[key]
    del self[key]
    return value

  def _construct(self, key: K) -> V:
    if self._callable_one_arg is not None:
      return self._callable_one_arg(key)
    elif self._callable_no_args is not None:
      return self._callable_no_args()
    raise ContractViolation("Somehow both callable is None?") # pragma: no cover
