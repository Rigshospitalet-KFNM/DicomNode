"""Script to generate gated pet images from multiple pet series / numpy images
"""

# Python standard library
from random import randint
from argparse import ArgumentParser, _SubParsersAction, Namespace
import json
from pathlib import Path
from typing import List

# Third party packages

# Dicomnode packages
from dicomnode.dicom import gen_uid
from dicomnode.dicom.series import DicomSeries
from dicomnode.data_structures.image_tree import DicomTree
from dicomnode.lib.io import discover_files, load_dicom, save_dicom

nifti_formats = [
  '.nii',
  '.nii.gz'
]

dicom_formats = [
  '.dcm',
  '.ima',
]

def sort_series_description(series_tree : DicomSeries):
  return series_tree[0x0008_103E]

sorting_algorithms = {
  'series_description' : sort_series_description
}

file_formats = nifti_formats + dicom_formats

def filterFormats(formats):
  def filterToDataFiles(path: Path):
    return any([path.name.endswith(file_format) for file_format in formats])
  return filterToDataFiles

def gatify_series(series: List[DicomSeries], config):
  """This Function does the actual construction

  This function is IO less, and should be the one that is actually tested.

  This function modifies series a


  Args:
      series (List[DicomSeries]): _description_
      config (_type_): _description_

  Raises:
      Exception: _description_
      Exception: _description_

  Returns:
      List[DicomSeries]: _description_
  """
  gate_num_images = len(series[0])

  for i, gate in enumerate(series):
    if len(gate) != gate_num_images:
      raise Exception(f"Gate: {i} has {len(gate)} images while other have {gate_num_images} images.")

  trigger_time = 0
  image_index = 0
  new_series_uid = gen_uid()
  new_series_number = randint(5000, 100000)
  for (gate, frame_time) in zip(series, config['frame_time']):
    if not isinstance(frame_time, float) and not isinstance(frame_time, int):
      raise Exception("One of the frame times is not a number!")

    image_indexes = [image_index + i for i in range(gate_num_images)]
    new_sop_uid = [gen_uid() for i in range(gate_num_images)]

    gate[0x0008_0018] = new_sop_uid

    gate[0x0018_1060] = trigger_time
    gate[0x0018_1063] = frame_time

    # R-R values
    gate[0x0018_1080] = 'N'

    gate[0x0020_000E] = new_series_uid
    gate[0x0020_0011] = new_series_number
    gate[0x0020_0013] = image_indexes

    gate[0x0054_0061] = 1
    gate[0x0054_0071] = len(series)
    gate[0x0054_1000] = 'GATED'

    gate[0x0054_1330] = image_indexes

    trigger_time += frame_time
    image_index += gate_num_images

  return series


def handle_dicom(paths: List[Path], config, output_destination: Path):
  datasets = [load_dicom(path) for path in paths]
  dicom_tree = DicomTree(datasets)

  if 'sorting' in config:
    sorting_algorithm = sorting_algorithms[config['sorting']]
  else:
    sorting_algorithm = sorting_algorithms['series_description']

  series = sorted([DicomSeries(series) for series in dicom_tree.series()],
                  key=sorting_algorithm)

  if 'frame_time' not in config :
    raise Exception("Frame times need to be in json config file")

  if len(series) != len(config['frame_time']):
    raise Exception("Need more frame times!")

  series = gatify_series(series=series, config=config)

  if not output_destination.exists():
    output_destination.mkdir()

  for gate in series:
    for dataset in gate:
      gate = dataset.ImageIndex // len(series)

      local_image_index = dataset.ImageIndex % dataset.NumberOfTimeSlots
      dataset_destination = output_destination / f"image_{local_image_index:05}_gate_{gate}.dcm"
      save_dicom(dataset_destination, dataset)

def handle_nifti(paths: List[Path], config, output_directory):
  pass


def get_parser(subparser: _SubParsersAction):
  _, _, tool_name = __name__.split(".")

  module_parser: ArgumentParser = subparser.add_parser(tool_name,
                       help="Constructs a Gated Pet series from numpy images or multiple pet series.")
  module_parser.add_argument('data_path', type=Path, help="Path to all your pet data")
  module_parser.add_argument('config_path', type=Path, help="json file with all configuration")
  module_parser.add_argument('output_directory', type=Path, help="path to directory")


def entry_func(args: Namespace):
  with open(args.config_path, 'r') as fp:
    config = json.load(fp)

  files = discover_files(args.data_path)
  dicom_files = list(filter(filterFormats(dicom_formats), files))
  nifti_files = list(filter(filterFormats(nifti_formats), files))

  is_dicom = len(dicom_files) > 0
  is_nifti = len(nifti_files) > 0

  if not is_dicom and not is_nifti:
    raise Exception("You need some data files to build a gated recon")
  if is_dicom and is_nifti :
    raise Exception("Cannot interweave dicom and nifti files")

  if is_dicom:
    handle_dicom(dicom_files, config, args.output_directory)
  if is_nifti:
    handle_nifti(nifti_files, config, args.output_directory)
