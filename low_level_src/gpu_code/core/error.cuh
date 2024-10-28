#pragma once
enum dicomNodeError_t: uint32_t {
  success = 0,
  NotLinearIndependant = 1,
  // Note that the specific cuda error is encoded
  cudaError = 0x80000000
};