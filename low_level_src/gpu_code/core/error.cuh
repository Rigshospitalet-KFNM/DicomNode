#pragma once

constexpr uint32_t cudaErrorFlag = 0x80000000;

enum dicomNodeError_t: uint32_t {
  SUCCESS = 0,
  NotLinearIndependant = 1,
  INPUT_TYPE_ERROR = 2,
  NON_POSITIVE_SHAPE = 3,

  // Note that the specific cuda error is encoded
  cudaError = cudaErrorFlag
};

static cudaError_t extract_cuda_error(dicomNodeError_t error){
  return (cudaError_t)(error ^ cudaErrorFlag);
}

static dicomNodeError_t encode_cuda_error(cudaError_t error){
  return (dicomNodeError_t)(error | cudaErrorFlag);
}

static bool is_cuda_error(dicomNodeError_t error){
  return error & cudaErrorFlag;
}