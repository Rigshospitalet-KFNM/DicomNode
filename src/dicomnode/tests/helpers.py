from pathlib import Path

from time import perf_counter
from typing import Any, Callable, Dict, Iterable,Iterator,List, Optional, Tuple, Type, Union

import cProfile
import pstats

import numpy
from pydicom import Dataset
from pydicom.uid import SecondaryCaptureImageStorage, UID

from dicomnode.lib.dicom import make_meta, gen_uid

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

    PerformanceDir = Path("performance").mkdir(exist_ok=True)
    stats.dump_stats(f"performance/{func.__name__}.prof")


    return ret
  return inner
