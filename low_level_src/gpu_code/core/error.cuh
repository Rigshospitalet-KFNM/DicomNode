#pragma once
#include<stdint.h>


constexpr uint32_t cudaErrorFlag = 0x80000000;

enum dicomNodeError_t: uint32_t {
  SUCCESS = 0,
  NotLinearIndependent = 1,
  INPUT_TYPE_ERROR = 2,
  NON_POSITIVE_SHAPE = 3,
  POINTER_IS_NOT_A_DEVICE_PTR = 4,
  UNABLE_TO_ACQUIRE_BUFFER = 5,
  INPUT_SIZE_MISMATCH = 6,

  // Note that the specific cuda error is encoded
  cudaError = cudaErrorFlag
};

static inline cudaError_t extract_cuda_error(const dicomNodeError_t error){
  return (cudaError_t)(error ^ cudaErrorFlag);
}

static inline dicomNodeError_t encode_cuda_error(const cudaError_t error){
  if (error) { return (dicomNodeError_t)(error | cudaErrorFlag); }
  return SUCCESS;
}

static inline bool is_cuda_error(const dicomNodeError_t error){
  return error & cudaErrorFlag;
}