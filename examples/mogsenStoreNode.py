
# Python Standard Library
import csv
from datetime import datetime
import logging
from os import environ
from pathlib import Path
from typing import Any, Dict, Iterable, List, Type

# Third Party Imports
from pydicom import Dataset

# Dicomnode Packages
from dicomnode.lib.io import save_dicom
from dicomnode.lib.dicom import make_meta
from dicomnode.server.input import AbstractInput
from dicomnode.server.nodes import AbstractPipeline
from dicomnode.server.pipeline_tree import InputContainer
from dicomnode.server.grinders import ListGrinder
from dicomnode.server.output import FileOutput


env_name = "STORENODE_ARCHIVE_PATH"
env_default = "/raid/dicom/storenode/"

ARCHIVE_PATH = environ.get(env_name, env_default)

INPUT_ARG: str = "dataset"

class DicomObjectInput(AbstractInput):
  required_tags: List[int] = [
    0x00080016, # SOPInstanceUID
    0x0020000D, # StudyInstanceUID
    0x0020000E, # SeriesInstanceUID
  ]

  required_values: Dict[int, Any]

  def validate(self):
    return True
  # End of input

def createSeriesFile(archive_path: Path, dataset: Dataset) -> None:
    dataset_path: Path = archive_path / dataset.PatientID / dataset.StudyInstanceUID.name / dataset.SeriesInstanceUID.name / (dataset.SOPInstanceUID.name + '.dcm')
    tsvfile_path: Path = archive_path / dataset.PatientID / dataset.StudyInstanceUID.name / (dataset.SeriesInstanceUID.name + '.tsv')

    # Initiate
    Slice = Frame = FrameDuration = 0

    # Different tags are used based on Modality
    if dataset.Modality == 'PT':
      # Get Series Type (0054,1000)
      SeriesType = dataset[0x00541000].value

      # Get Image Index
      ImageIndex = dataset[0x00541330].value

      # Determine slice and possibly frame number from Image Index
      # Different behaviour based on SeriesType [STATIC, DYNAMIC, GATED, WHOLE BODY]
      if 'DYNAMIC' in SeriesType:
        # Get number of slices
        NumberOfSlices = dataset[0x00540081].value

        # FROM DICOM MANUAL
        # ImageIndex = ((Time Slice Index - 1) * (NumberOfSlices)) + Slice Index
        Frame, Slice = divmod(ImageIndex-1,NumberOfSlices)
        Frame += 1
        Slice += 1
      elif any(substring in SeriesType for substring in ['STATIC','WHOLE BODY']):
        Slice = ImageIndex
        Frame = 1

      # Get actual frame duration in seconds or set to zero
      FrameDuration = dataset[0x00181242].value * 0.001
      if FrameDuration.is_integer():
          FrameDuration = int(FrameDuration)

    elif dataset.Modality == 'CT':
        # Use Instance Number (0020, 0013) as Slice (Not guarenteed to be spatially sorted)
        Slice = dataset.get(0x00200013,0).value

    # Create Series Acquisition timestamp from acquisition date (0008, 0022) and time (0008, 0032)
    AcquisitionTime = datetime.strptime(dataset[0x00080022].value+dataset[0x00080032].value,'%Y%m%d%H%M%S.%f')

    # Create Series timestamp from series date (0008, 0021) and series time (0008, 0031)
    SeriesTime = datetime.strptime(dataset[0x00080021].value+dataset[0x00080031].value,'%Y%m%d%H%M%S.%f')

    # Subtract to get Frame Start Time
    FrameTimeStart = (AcquisitionTime - SeriesTime).total_seconds()

    # Convert to integer if already integer in floating point
    if FrameTimeStart.is_integer():
      FrameTimeStart = int(FrameTimeStart)

    # Create data row to insert in .tsv file
    data = [Slice, Frame, FrameTimeStart, FrameDuration, AcquisitionTime.strftime('%H:%M:%S'), dataset_path]

    # Header fields for .tsv file
    header = ['Plane', 'Frame', 'Start', 'Dur', 'AcqTime', 'File']

    # Check if file is being created then write header
    if not Path(tsvfile_path).is_file():
      write_header = True
    else:
      write_header = False

    # Open for appending
    with open(tsvfile_path, 'a', encoding='UTF8') as tsvfile:
      writer = csv.writer(tsvfile, delimiter='\t')

      # write the header
      if write_header:
        writer.writerow(header)

      # write the data
      writer.writerow(data)

    return None

def storeDataset(archive_path: Path, dataset: Dataset) -> None:
    dataset_path: Path = archive_path / dataset.PatientID / dataset.StudyInstanceUID.name / dataset.SeriesInstanceUID.name / (dataset.SOPInstanceUID.name + '.dcm')

    if not Path(dataset_path).is_file():
      save_dicom(dataset_path,  dataset)
      createSeriesFile(archive_path, dataset)

class StoreNode(AbstractPipeline):
  log_path: str = "/var/log/storenode.log"
  ae_title: str = "STORENODE"
  port: int = 1337
  ip: str = '0.0.0.0'
  disable_pynetdicom_logger=True
  log_level: int = logging.INFO
  patient_identifier_tag = 0x0020_000E # SeriesInstanceUID
  #patient_identifier_tag = 0x00100020 # PatientID

  input: Dict[str, Type] = {
    INPUT_ARG : DicomObjectInput
  }

  archive_path: Path = Path(ARCHIVE_PATH)

  def process(self, input_data: InputContainer) -> FileOutput:
    #create tab file
    self.logger.info(f"Started to store {len(input_data[INPUT_ARG])}")
    datasets: List [Dataset] = input_data[INPUT_ARG]
    return FileOutput([(self.archive_path, datasets)], storeDataset)

  def post_init(self) -> None:
    self.archive_path.mkdir(exist_ok=True)


if __name__ == "__main__":
   node = StoreNode()
   node.open()
