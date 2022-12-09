from pydicom import Dataset
from pydicom.uid import SecondaryCaptureImageStorage

from typing import Iterator
from dicomnode.lib.dicom import make_meta, gen_uid

import numpy

def generate_numpy_dataset(
    StudyUID,
    SeriesUID,
    Cols,
    Rows,
    bits,
    rescale
  ) -> Dataset:
  ds = Dataset()
  ds.SOPClassUID = SecondaryCaptureImageStorage

  ds.SOPInstanceUID = gen_uid()
  ds.StudyInstanceUID = StudyUID
  ds.SeriesInstanceUID = SeriesUID

  ds.Rows = Rows
  ds.Columns = Cols

  if bits % 8 == 0:
    ds.BitsAllocated = bits
  else:
    ds.BitsAllocated = bits + (8 - (bits % 8))
  ds.BitsStored = bits
  ds.HighBit = bits - 1

  if rescale:
    slope = numpy.random.uniform(0, 2 / (2 ** - 1))
    intercept = numpy.random.uniform(0, (2 ** - 1))
    ds.RescaleSlope = slope
    ds.RescaleIntercept = intercept

  image = numpy.random.randint(0, 2 ** bits - 1, (Cols, Rows))
  ds.PixelData = image.tobytes()


  return ds

def generate_numpy_datasets(
    datasets: int,
    StudyUID = gen_uid(),
    SeriesUID = gen_uid(),
    Cols = 400,
    Rows = 400,
    Bits = 16,
    rescale = True
  ) -> Iterator[Dataset]:
  yielded = 0
  while yielded < datasets:
    yielded += 1
    yield generate_numpy_dataset(
      StudyUID,
      SeriesUID,
      Cols,
      Rows,
      Bits,
      rescale
    )

