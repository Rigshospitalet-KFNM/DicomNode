#pragma once
#include<stdint.h>


constexpr uint32_t cudaErrorFlag = 0x80000000;


/**
 * @brief This is the enum of all the errors that can happen using this library
 * Note that this is not an exhaustive list of valid dicomnodeError_t values
 * as this enum also cover cudaError_t. So if the program have encountered a
 * cudaErrorInvalidValue (which have value 1) then this will take a value of
 * 0x80000001
 */
enum dicomNodeError_t: uint32_t {
  SUCCESS = 0,
  NotLinearIndependent = 1,
  INPUT_TYPE_ERROR = 2,
  NON_POSITIVE_SHAPE = 3,
  POINTER_IS_NOT_A_DEVICE_PTR = 4,
  UNABLE_TO_ACQUIRE_BUFFER = 5,
  INPUT_SIZE_MISMATCH = 6,
  ALREADY_ALLOCATED_OBJECT = 7, // This is triggered if you try to allocate to an object that's already alocated

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