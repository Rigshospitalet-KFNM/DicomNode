import dataclasses

class DataInput:
  pass


class AbstractInputDataClass:
  pass

  @classmethod
  def validate(cls) -> bool:
    return False