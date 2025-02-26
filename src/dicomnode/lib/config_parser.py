import re
from pathlib import Path
from typing import Any, Dict,get_args,get_origin, List, Optional,TypeAlias, Type, Union, _GenericAlias # type: ignore # WHY IS THIS NOT PUBLIC?
import warnings

LIST_REGEX = re.compile(r"\[(.*)\]")

def _get_config_resource(config_resource_handle) -> Dict[str, Any]:
  if isinstance(config_resource_handle, str):
    config = Path(config_resource_handle)

  return {}

def _typecast_value_to_list(value, annotation: List) -> Optional[List]:
  args = get_args(annotation)

  if isinstance(value, List):
    return value

  if isinstance(value, str) and 0 < len(args):
    target_type, = args
    regex_match = LIST_REGEX.match(value)
    if regex_match is None:

      return None

    list_content, = regex_match.groups()

    return [ target_type(substring.strip()) for substring in list_content.split(',') ]

  print("Missing all the checks")

  raise TypeError()

def _typecast_value_to_union(value, annotation):
  args = get_args(annotation)

  none_type_in_args = type(None) in args

  if none_type_in_args:
    if isinstance(value, str) and value.lower() == "none" or value is None:
      return None

  for arg in args:
    try:
      type_casted = arg(value)
      return type_casted
    except TypeError:
      pass
    except ValueError:
      pass

  raise TypeError


# FUCK I Need exceptions to handle None
def _typecast_generic_alias(value, annotation):
  origin = get_origin(annotation)

  if origin is list or origin is List:
    return _typecast_value_to_list(value, annotation)

  if origin is Union:
    return _typecast_value_to_union(value, annotation)



  raise TypeError(f"Unable to convert to {annotation}")

def _typecast_config(raw_config: Dict[str, Any], annotations: Dict[str, Any]):
  """_summary_

  Args:
      config (_type_): _description_
      annotations (_type_): _description_

  Returns:
      _type_: _description_
  """

  """So I

  Returns:
      _type_: _description_
  """
  return_config = {}

  for key, value in raw_config.items():
    if key not in annotations:
      warnings.warn(f"The {key} in config, was not found in the annotation\n"
                    "Please Annotate the attribute with : type(attribute)"
                    , stacklevel=1000)
      continue

    annotation = annotations[key]
    if isinstance(annotation, type):
      return_config[key] = annotation(value)
      continue
    elif isinstance(annotation, _GenericAlias):
      try:
        return_config[key] = _typecast_generic_alias(value, annotation)
      except TypeError:
        warnings.warn(f"At config key: {key} Unable to create a {annotation} from"
                      f" {value} of type {type(value)}"
                      , stacklevel=1000)
      continue
    print(f"Missed annotation of {annotation}")


  return return_config

def get_config(
    config_resource_handle,
    annotations: Dict[str, Any]) -> Dict[str, Any]:
  """_summary_

  Args:
      config_resource_handle (_type_): _description_
      annotations (Dict[str, Any]): _description_

  Returns:
      Dict[str, Any]: _description_
  """
  """This function does 2 things and delegates them:
  1. Get the config from the resource, if it fails return {}
  2. typecast the config to the annotations.
  """


  config = _get_config_resource(config_resource_handle)

  return _typecast_config(config, annotations)