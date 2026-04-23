# Python standard library
from unittest import skipUnless

# Third party libraries

# Dicomnode modules
from dicomnode.lib.exceptions import GPUError, ContractViolation
from dicomnode.math import CUDA
from dicomnode.math.gpu_helpers import gpu_call

# Tests Helpers
from tests.helpers.dicomnode_test_case import DicomnodeTestCase

class GPUHelperTestCases(DicomnodeTestCase):
  @skipUnless(CUDA, "You need GPU")
  def test_gpu_helper_success_function(self):
    from dicomnode.math._cuda import DicomnodeError

    def simulated_gpu_function():
      return DicomnodeError(0)

    result = gpu_call(simulated_gpu_function)
    self.assertIsNone(result)

  @skipUnless(CUDA, "You need GPU")
  def test_gpu_helper_success_function_tuple(self):
    from dicomnode.math._cuda import DicomnodeError

    def simulated_gpu_function():
      return DicomnodeError(0), "Hello world"

    result = gpu_call(simulated_gpu_function)
    self.assertEqual(result, "Hello world")

  @skipUnless(CUDA, "You need GPU")
  def test_gpu_helper_gpu_error_raises(self):
    from dicomnode.math._cuda import DicomnodeError

    def simulated_gpu_function():
      return DicomnodeError(1)

    self.assertRaises(GPUError, gpu_call, simulated_gpu_function)

  @skipUnless(CUDA, "You need GPU")
  def test_gpu_helper_gpu_error_raises_tuple(self):
    from dicomnode.math._cuda import DicomnodeError

    def simulated_gpu_function():
      return DicomnodeError(1), None

    self.assertRaises(GPUError, gpu_call, simulated_gpu_function)

  @skipUnless(CUDA, "You need GPU")
  def test_gpu_helper_contract_violation_raises(self):
    from dicomnode.math._cuda import DicomnodeError

    def simulated_gpu_function():
      return None, None

    self.assertRaises(ContractViolation, gpu_call, simulated_gpu_function)
