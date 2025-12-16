#pragma once
#include<stdint.h>
#include<string>
#include<sstream>
#include<iostream>

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
  NOT_LINEAR_INDEPENDENT = 1,
  INPUT_TYPE_ERROR = 2,
  NON_POSITIVE_SHAPE = 3,
  POINTER_IS_NOT_A_DEVICE_PTR = 4,
  UNABLE_TO_ACQUIRE_BUFFER = 5,
  INPUT_SIZE_MISMATCH = 6,
  ALREADY_ALLOCATED_OBJECT = 7, // This is triggered if you try to allocate to an object that's already alocated
  ARG_IS_NULL_POINTER = 8,
  // Note that the specific cuda error is encoded
  cudaError = cudaErrorFlag
};

static inline cudaError_t extract_cuda_error(const dicomNodeError_t error) noexcept{
  return (cudaError_t)(error ^ cudaErrorFlag);
}

static inline dicomNodeError_t encode_cuda_error(const cudaError_t error) noexcept{
  if (error) { return (dicomNodeError_t)(error | cudaErrorFlag); }
  return SUCCESS;
}

static inline bool is_cuda_error(const dicomNodeError_t error) noexcept{
  return error & cudaErrorFlag;
}

static std::string error_to_human_readable(const dicomNodeError_t error) noexcept {
  if(is_cuda_error(error)){
    return cudaGetErrorName(extract_cuda_error(error));
  }

  switch (error){
    case NON_POSITIVE_SHAPE:
      return "None positive shape of extent";
    case NOT_LINEAR_INDEPENDENT:
      return "Not linear independent matrix";
    case INPUT_SIZE_MISMATCH:
      return "Input sizes mismatching";
    case ALREADY_ALLOCATED_OBJECT:
      return "Already allocated object";
    default:
      const std::string error_message = "unknown error of code: "
          + std::to_string(static_cast<uint32_t>(error));
      return error_message;
  }
}