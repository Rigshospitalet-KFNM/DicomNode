"""Script to generate gated pet images from multiple pet series / numpy images
"""

# Python standard library
from random import randint
from argparse import ArgumentParser, _SubParsersAction, Namespace
import json
from math import log10, floor
from pathlib import Path
from typing import List

# Third party packages
from pydicom.tag import Tag
from pydicom.datadict import dictionary_VR
from pydicom.uid import PositronEmissionTomographyImageStorage
from nibabel.loadsave import load as load_nifti

# Dicomnode packages
from dicomnode.dicom import gen_uid
from dicomnode.dicom.dicom_factory import Blueprint, DicomFactory, StaticElement

from dicomnode.dicom.series import DicomSeries, NiftiSeries
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
  return series_tree[0x0008_103E].value # type: ignore

sorting_algorithms = {
  'series_description' : sort_series_description
}

FRAME_TIME_KEY = "frame_time"
NOMINAL_INTERVAL_KEY = "nominal_interval"
LOW_RR_VALUE_KEY = "low_rr_value"
HIGH_RR_VALUE_KEY = "high_rr_value"
INTERVALS_ACQUIRED_KEY = "intervals_acquired"
INTERVALS_REJECTED_KEY = "intervals_rejected"
BEAT_REJECTION_FLAG_KEY = "beat_rejection_flag"
CARDIAC_FRAMING_TYPE_KEY = "cardiac_framing_type"
SKIP_BEATS_KEY = "skip_beats"
HEART_RATE_KEY = "heart_rate"

required_config_tags = [
  FRAME_TIME_KEY,
  NOMINAL_INTERVAL_KEY,
  LOW_RR_VALUE_KEY,
  HIGH_RR_VALUE_KEY,
  INTERVALS_ACQUIRED_KEY,
  INTERVALS_REJECTED_KEY,
  BEAT_REJECTION_FLAG_KEY,
  CARDIAC_FRAMING_TYPE_KEY,
  SKIP_BEATS_KEY,
  HEART_RATE_KEY,
]


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

  frame_of_reference_uid = gen_uid()

  trigger_time = 0
  image_index = 0
  new_series_uid = gen_uid()
  new_series_number = randint(5000, 100000)

  for (gate, frame_time) in zip(series, config[FRAME_TIME_KEY]):
    if not isinstance(frame_time, float) and not isinstance(frame_time, int):
      raise Exception("One of the frame times is not a number!")

    image_indexes = [image_index + i + 1 for i in range(gate_num_images)]
    new_sop_uid = [gen_uid() for _ in range(gate_num_images)]

    gate["NumberOfSlices"] = gate_num_images

    gate["SOPInstanceUID"] = new_sop_uid # SOPInstanceUID

    gate["TriggerTime"] = trigger_time # Trigger Time
    #gate[0x0018_1061] = "EKG"
    gate["NominalInterval"] = config[NOMINAL_INTERVAL_KEY]
    gate["FrameTime"] = frame_time # Frame time
    gate["CardiacFramingType"] = config[CARDIAC_FRAMING_TYPE_KEY]

    # R-R values
    gate["BeatRejectionFlag"] = config[BEAT_REJECTION_FLAG_KEY]
    gate["LowRRValue"] = config[LOW_RR_VALUE_KEY]
    gate["HighRRValue"] = config[HIGH_RR_VALUE_KEY]
    gate["IntervalsAcquired"] = config[INTERVALS_ACQUIRED_KEY]
    gate["IntervalsRejected"] = config[INTERVALS_REJECTED_KEY]
    gate["SkipBeats"] = config[SKIP_BEATS_KEY]
    gate["HeartRate"] = config[HEART_RATE_KEY]

    gate["SeriesInstanceUID"] = new_series_uid
    gate["SeriesNumber"] = new_series_number
    gate["InstanceNumber"] = image_indexes # Instance Numbers

    #gate[0x0020_0052] = frame_of_reference_uid # Frame of Reference UID

    gate["NumberOfRRIntervals"] = 1 # Number of R-R Intervals
    gate["NumberOfTimeSlots"] = len(series) # Number of Time Slots
    gate["SeriesType"] = ['GATED', "IMAGE"] # Series type
    gate["ImageIndex"] = image_indexes # image Index

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

  series = [DicomSeries([ds for ds in series]) for series in dicom_tree.series()]
  series = sorted(series, key=sorting_algorithm) # type: ignore

  if 'frame_time' not in config :
    raise Exception("Frame times need to be in json config file")

  if len(series) != len(config['frame_time']):
    raise Exception(f"There's {len(series)} Series, and {len(config['frame_time'])} frame_times!")

  series = gatify_series(series=series, config=config)

  if not output_destination.exists():
    output_destination.mkdir()

  for gate in series:
    for dataset in gate:
      zImageIndex = dataset.ImageIndex - 1

      gate = zImageIndex  // (dataset.NumberOfSlices)
      local_image_index = (zImageIndex % dataset.NumberOfSlices) +1
      dataset_destination = output_destination / f"gate_{gate + 1}_image_{local_image_index:03}.dcm"
      save_dicom(dataset_destination, dataset)

BASE_BLUEPRINT = Blueprint([
  StaticElement(0x0008_0005, 'CS', 'ISO_IR 100'),
  StaticElement(0x0008_0005, 'CS', ['ORIGINAL', 'PRIMARY']),
  StaticElement(0x0008_0016, 'UI', PositronEmissionTomographyImageStorage),
])


def get_blueprint_for_nifti(i, config, nifti_series, num_series):
  if 'dicom_tags' not in config:
    raise Exception("You need dicom_tags in your config to construct dicom series from nifti")

  blueprint = Blueprint(BASE_BLUEPRINT)

  for str_tag, value in config['dicom_tags']:
    tag = Tag(str_tag)
    vr = dictionary_VR(tag)

    if isinstance(value, List) and len(value) == num_series:
      element = StaticElement(tag, vr, value[i])
    else:
      element = StaticElement(tag, vr, value)

    blueprint.add_virtual_element(element)

  return blueprint


def handle_nifti(paths: List[Path], config, output_directory):
  nifti_s = [load_nifti(path) for path in paths]

  series = [NiftiSeries(nifti) for nifti in nifti_s] # type: ignore

  factory = DicomFactory()

  blueprints = [
    get_blueprint_for_nifti(i, config, series_, len(series)) for i, series_ in enumerate(series)
  ]

  dicom_series = [
    factory.build_nifti_series(series_, blueprint, {"series" : i}) for i, (series_, blueprint) in enumerate(zip(series, blueprints))
  ]

  for gate_num, dicom_series in enumerate(dicom_series):
    for image_num, dicom_slice in enumerate(dicom_series):
      image_index = image_num + 1
      dataset_destination = output_directory / f"gate_{gate_num + 1}_image_{image_index}.dcm"
      save_dicom(dataset_destination, dicom_slice)


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
