from pathlib import Path
from typing import Optional
import os

import nibabel
from nibabel import processing
from nibabel.filebasedimages import FileBasedImage
import dicomnode

os.environ["TEMPLATEFLOW_HOME"] = str(dicomnode.library_paths.report_data_directory)

try:
  import templateflow
  client = templateflow.TemplateFlowClient(dicomnode.library_paths.report_data_directory)

except ImportError as exception:
  import warnings
  warnings.warn("Optional Dependency is missing - to install pip install templateflow ")
  raise exception

MNI152_KW = "MNI152Lin"


def get_MNI152(suffix="T1w", resolution: int = 1) -> Optional[FileBasedImage]:
  if resolution not in [1,2]:
    raise FileNotFoundError("Only resolution 1,2 is provided by template flow")

  tpl_directory = dicomnode.library_paths.report_data_directory / "tpl-MNI152Lin"

  file = tpl_directory / f"tpl-MNI152Lin_res-0{resolution}_{suffix}.nii.gz"

  if file.exists():
    return nibabel.loadsave.load(file)


  # Template flow is really really really slow
  paths = client.get(MNI152_KW, suffix=suffix, resolution=resolution, extension='nii.gz') # type: ignore

  if isinstance(paths, Path):
    return nibabel.loadsave.load(paths)
  else:
    if len(paths) == 0:
      print("FOUND NOTHING")
      return None

    return nibabel.loadsave.load(paths[0])
