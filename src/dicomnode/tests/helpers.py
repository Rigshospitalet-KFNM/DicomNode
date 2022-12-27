import cProfile
import pstats
from logging import Logger
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional,\
                    Tuple, Type, Union

import numpy
from pydicom import Dataset
from pydicom.uid import UID, SecondaryCaptureImageStorage
from pynetdicom import events
from pynetdicom.ae import ApplicationEntity

from dicomnode.lib.dicom import gen_uid, make_meta

unsigned_array_encoding: Dict[int, Type[numpy.unsignedinteger]] = {
  8 : numpy.uint8,
  16 : numpy.uint16,
  32 : numpy.uint32,
  64 : numpy.uint64,
}

def generate_numpy_dataset(
    StudyUID: UID,
    SeriesUID: UID,
    Cols: int,
    Rows: int,
    bits: int,
    rescale: bool,
    PixelRepresentation: int,
    PatientID: str
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

  make_meta(ds)

  if bits % 8 == 0:
    ds.BitsAllocated = bits
  else:
    ds.BitsAllocated = bits + (8 - (bits % 8))
  ds.BitsStored = bits
  ds.HighBit = bits - 1

  dType = unsigned_array_encoding.get(ds.BitsAllocated)

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
    PatientID: str = "None"
  ) -> Iterator[Dataset]:
  yielded = 0
  while yielded < datasets:
    yielded += 1
    ds = generate_numpy_dataset(
      StudyUID,
      SeriesUID,
      Cols,
      Rows,
      Bits,
      rescale,
      PixelRepresentation,
      PatientID
    )
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
    Path("performance").mkdir(exist_ok=True)
    stats.dump_stats(f"performance/{func.__name__}.prof")
    return ret
  return inner


def get_test_ae(port: int, destination_port:int, logger: Logger, dataset: Optional[Dataset] = None):
  if dataset is None:
    dataset = Dataset()
    dataset.SOPInstanceUID = gen_uid()
    dataset.SeriesInstanceUID = gen_uid()

  def _handle_C_store(evt: events.Event):
    logger.info("Received C Store")
    return 0x0000

  def _handle_C_move(evt: events.Event):
    logger.info("Received C Move")
    identifier = evt.identifier # Dataset send by c move

    # yield destination ip address and port
    yield 'localhost', destination_port
    # yield number of C-stores
    yield 1

    # For dataset in container:
    #    if Cancelled
    #       yield 0xFE00, None
    #    yield (0xFF00,dataset)
    if evt.is_cancelled:
      yield 0xFE00, None

    yield 0xFF00, Dataset

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

  ae = ApplicationEntity(ae_title="Dummy")

  ae.start_server(('localhost', port),
    evt_handlers=[
      (events.EVT_C_MOVE, _handle_C_move),
      (events.EVT_C_STORE, _handle_C_store),
      (events.EVT_C_FIND, _handle_C_find),
      (events.EVT_C_GET, _handle_C_get)
    ],
    block=False
  )

  return ae
