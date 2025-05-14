from dicomnode import math

if math.CUDA:
  from dicomnode.math import _cuda

  def _gpu_labeling(image: 'math.image.Image'):
    return _cuda.labeling.slice_based(image)
