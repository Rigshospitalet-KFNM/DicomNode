

class Counter:
  "Single Threaded lock"
  def __init__(self) -> None:
    self._count = 0

  def increment(self):
    self._count += 1

  def get(self):
    return self._count