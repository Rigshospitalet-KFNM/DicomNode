# Python standard library
import cProfile
from functools import wraps
import os
from logging import Logger
from pathlib import Path
import pstats
import threading
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional,\
                    Tuple, Type, Union
from unittest import TestCase

# Third party Packages
import numpy
from pydicom import Dataset
from pydicom.uid import UID, SecondaryCaptureImageStorage
from pynetdicom import events
from pynetdicom.ae import ApplicationEntity
from pynetdicom.presentation import AllStoragePresentationContexts, build_context
from pynetdicom.sop_class import StudyRootQueryRetrieveInformationModelFind, StudyRootQueryRetrieveInformationModelMove, PatientRootQueryRetrieveInformationModelFind, PatientRootQueryRetrieveInformationModelMove  #type: ignore
from psutil import Process

# Dicomnode Packages
# Test helpers
try:
  TESTING_TEMPORARY_DIRECTORY = os.environ['DICOMNODE_TESTING_TEMPORARY_DIRECTORY']
except KeyError:
  TESTING_TEMPORARY_DIRECTORY = "/tmp/dicomnode_tests"

# Dicomnode
from dicomnode.constants import UNSIGNED_ARRAY_ENCODING
from dicomnode.lib.logging import set_logger
from dicomnode.dicom import gen_uid, make_meta

def _generate_numpy_dataset(
    StudyUID: UID,
    SeriesUID: UID,
    Cols: int,
    Rows: int,
    bits: int,
    rescale: bool,
    PixelRepresentation: int,
    PatientID: str,
    image_position,
    image_orientation,
    pixel_spacing,
    slice_thickness,

  ) -> Dataset:
  ds = Dataset()
  ds.SOPClassUID = SecondaryCaptureImageStorage
  ds.PatientID = PatientID
  ds.SOPInstanceUID = gen_uid()
  ds.StudyInstanceUID = StudyUID
  ds.SeriesInstanceUID = SeriesUID

  ds.PhotometricInterpretation = "MONOCHROME2"
  ds.SamplesPerPixel = 1
  ds.Rows = Rows
  ds.Columns = Cols
  ds.PixelRepresentation = PixelRepresentation

  ds.ImagePositionPatient = image_position
  ds.ImageOrientationPatient = image_orientation
  ds.PixelSpacing = pixel_spacing
  ds.SliceThickness = slice_thickness

  make_meta(ds)

  if bits % 8 == 0:
    ds.BitsAllocated = bits
  else:
    ds.BitsAllocated = bits + (8 - (bits % 8))
  ds.BitsStored = bits
  ds.HighBit = bits - 1

  dType = UNSIGNED_ARRAY_ENCODING.get(ds.BitsAllocated)

  if dType is None:
    raise ValueError

  if rescale:
    slope = numpy.random.uniform(0, 2 / (2 ** bits - 1))
    intercept = numpy.random.uniform(0, (2 ** bits - 1))
    ds.RescaleSlope = slope
    ds.RescaleIntercept = intercept

  image = numpy.random.randint(0, 2 ** bits - 1, (Cols, Rows), dtype=dType)
  ds.PixelData = image.tobytes()

  return ds

def generate_numpy_datasets(
    datasets: int,
    StudyUID = gen_uid(),
    SeriesUID = gen_uid(),
    Cols = 400,
    Rows = 400,
    Bits = 16,
    rescale = True,
    PixelRepresentation = 0,
    PatientID: str = "None",
    starting_image_position = None,
    image_orientation = None,
    pixel_spacing = None,
    slice_thickness = 4
  ) -> Iterator[Dataset]:
  if starting_image_position is None:
    image_position = [0,0,0]
  else:
    image_position = starting_image_position
  if image_orientation is None:
    image_orientation = [1,0,0,0,1,0]
  if pixel_spacing is None:
    pixel_spacing = [4,4]

  yielded = 0
  while yielded < datasets:
    yielded += 1
    ds = _generate_numpy_dataset(
      StudyUID,
      SeriesUID,
      Cols,
      Rows,
      Bits,
      rescale,
      PixelRepresentation,
      PatientID,
      image_position,
      image_orientation,
      pixel_spacing,
      slice_thickness
    )
    image_position[2] += slice_thickness

    ds.InstanceNumber = yielded + 1
    yield ds

def generate_id_dataset(
    datasets: int,
    StudyUID = gen_uid(),
    SeriesUID = gen_uid(),
    PatientID = "1233435567"
  ):

  yielded = 0
  while yielded < datasets:
    yielded += 1
    dataset = Dataset()
    dataset.add_new(0x0020000D,'UI',StudyUID)
    dataset.add_new(0x0020000E,'UI',SeriesUID)
    dataset.add_new(0x00080016,'UI',gen_uid())
    dataset.add_new(0x00100020, 'LO', PatientID)
    yield dataset

def personify(
    tags: List[Tuple[int, str, Any]] = []
  ) -> Callable[[Dataset],None]:
  def retFunc(ds: Dataset) -> None:
    for tag,vr, val in tags:
      ds.add_new(tag, vr, val)
    return None
  return retFunc

def bench(func: Callable) -> Callable:
  """Decorator that benchmarks a function to stdout
  """
  def inner(*args, **kwargs) -> Any:
    with cProfile.Profile() as profile:
      ret = func(*args, **kwargs)

    stats = pstats.Stats(profile)
    stats.sort_stats(pstats.SortKey.TIME)
    Path("/tmp/performance").mkdir(exist_ok=True)
    stats.dump_stats(f"/tmp/performance/{func.__name__}.prof")
    return ret
  return inner


def get_test_ae(port: int, destination_port:int, logger: Logger, dataset: Optional[Dataset] = None):
  if dataset is None:
    dataset = Dataset()
    dataset.SOPInstanceUID = gen_uid()
    dataset.SeriesInstanceUID = gen_uid()
    dataset.StudyInstanceUID = gen_uid()
    dataset.SOPClassUID = SecondaryCaptureImageStorage
    dataset.PatientID = "1506932263"
    make_meta(dataset)

  def _handle_C_store(evt: events.Event):
    logger.info("Received C Store")

    return 0x0000

  def _handle_C_move(evt: events.Event):
    logger.info("Received C Move")
    identifier = evt.identifier # Dataset send by c move
    reqested_contexts = [build_context(identifier.SOPClassUID)]

    # yield destination ip address and port
    kwargs = {
      "contexts": reqested_contexts
    }

    yield ('127.0.0.1', destination_port, kwargs)
    # yield number of C-stores

    number_of_datasets = 1

    yield number_of_datasets

    for dataset_index in range(number_of_datasets):
      if evt.is_cancelled:
        yield 0xFE00, None

      yield 0xFF00, dataset

    logger.info("Finished handling C Move")

  def _handle_C_find(evt: events.Event):
    if 'QueryRetrieveLevel' not in evt.identifier:
      return 0xC000, None

    if evt.is_cancelled:
      yield 0xFE00, None

    yield 0xFF00, dataset


  def _handle_C_get(evt: events.Event):
    if 'QueryRetrieveLevel' not in evt.identifier:
      return 0xC000, None

    if evt.is_cancelled:
      yield 0xFE00, None

    yield 0xFF00, dataset

  ae = ApplicationEntity()
  ae.supported_contexts = AllStoragePresentationContexts
  ae.add_supported_context(StudyRootQueryRetrieveInformationModelFind)
  ae.add_supported_context(StudyRootQueryRetrieveInformationModelMove)
  ae.add_supported_context(PatientRootQueryRetrieveInformationModelMove)
  ae.add_supported_context(PatientRootQueryRetrieveInformationModelFind)
  ae.start_server(('127.0.0.1', port),
    evt_handlers=[
      (events.EVT_C_MOVE, _handle_C_move),
      (events.EVT_C_STORE, _handle_C_store),
      (events.EVT_C_FIND, _handle_C_find),
      (events.EVT_C_GET, _handle_C_get)
    ],
    block=False
  )

  return ae

def testing_logs():
  """Set or reset logs up for testing"""
  set_logger(None)

def process_thread_check_leak(func):
  @wraps(func)
  def wrapper(*args, **kwargs):
    testcase = args[0]

    if not isinstance(testcase, TestCase):
      raise TypeError(f"Invalid usage of process_thread_check_leak decorator on {func.__name__}")

    start_threads = threading.enumerate()

    return_value = func(*args, **kwargs)

    end_threads = threading.enumerate()

    for thread in end_threads:
      if thread not in start_threads:
        testcase.fail(f"Leaked a thread {thread.name} from {testcase._testMethodName}")

    return return_value

  return wrapper


from . import dicomnode_test_case