"""Tool for """

# Python standard library
from argparse import ArgumentParser, _SubParsersAction, Namespace
from pathlib import Path

# Third party packages
import matplotlib

# Dicomnode packages
from dicomnode.lib.exceptions import MissingDatasets
from dicomnode.lib.io import load_dicoms
from dicomnode.dicom.series import DicomSeries
from dicomnode.dicom.visualization import track_image_to_axes, Orientation
from dicomnode.math.image import Image

def get_parser(subparser: _SubParsersAction):
  _, _, tool_name = __name__.split(".")
  module_parser: ArgumentParser = subparser.add_parser(
    tool_name,
    help="Displays a dicom series in an interactive matplotlib figure, that you can scroll through.")
  module_parser.add_argument(
    'dicom_dir',
    type=Path,
    help="Path to directory containing the dicom files you wish to display")
  module_parser.add_argument(
    '--orientation', choices=['x', 'y', 'z',], default='z',
    help="Which direction which you want to scroll through\n"\
         "  x - the sagittal plane\n"\
         "  y - the coronal plane\n"\
         "  z - transverse plane"
  )


def entry_func(args: Namespace):
  datasets = load_dicoms(args.dicom_dir)

  match args.orientation.lower():
    case 'x':
      orientation = Orientation.X
    case 'y':
      orientation = Orientation.Y
    case 'z':
      orientation = Orientation.Z
    case _:
      raise Exception(f"Unknown Orientation: {args.orientation}")

  matplotlib.use('TkAgg') # This should be an argument
  import matplotlib.pyplot as plt
  plt.ion()

  if len(datasets) == 0:
    error_message = f"No datasets found at {args.dicom_dir}"
    raise MissingDatasets(error_message)

  series = DicomSeries(datasets)

  fig, ax = plt.subplots(1,1)

  tracker = track_image_to_axes(fig, ax, series.image, orientation=orientation)

  fig.show()
  fig.canvas.start_event_loop()
