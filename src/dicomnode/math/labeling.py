from dicomnode.math import CUDA
from dicomnode.math.image import Image

if CUDA:
  from dicomnode.math import _cuda

  def _gpu_labeling(image: Image):
    return _cuda.labeling.slice_based(image)
