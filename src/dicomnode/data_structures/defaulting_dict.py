from typing import Callable, Dict


class DefaultingDict[K,V]:
  """This is a dict that default constructs an element when you try and get it.
  The main idea is that you retrieve and mutate the elements of the Dict.
  """
  def __init__(self, callable_: Callable[[], V]) -> None:
    self._dict: Dict[K,V] = {}
    self.callable = callable_

  def __getitem__(self, key: K) -> V:
    if key not in self._dict:
      self._dict[key] = self.callable()

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