# Python standard library
from unittest import TestCase

# Third party modules

# Dicomnode modules
from dicomnode.math.types import PerformanceException, CudaException,\
    CudaErrorEnum

from tests.helpers.dicomnode_test_case import DicomnodeTestCase

class ExceptionsTestCase(DicomnodeTestCase):
  def test_normal_performance_exceptions(self):
    """Normal Performance exceptions are not fatal as they do not interact
    with a driver that can die"""
    self.assertFalse(PerformanceException().fatal)

  def test_cuda_exceptions(self):
    self.assertTrue(CudaException(CudaErrorEnum.cudaErrorAssert).fatal)
    self.assertFalse(CudaException(CudaErrorEnum.cudaErrorInvalidClusterSize).fatal)
