#include <gtest/gtest.h>
#include"../gpu_code/dicom_node_gpu.cuh"

#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>

#include<iostream>

TEST(CUDA_MANAGEMENT_TESTS, Free_Device_memory) {
  int* ptr;
  const cudaError_t error = cudaMalloc(&ptr, sizeof(int) * 10);
  EXPECT_EQ(error, cudaSuccess);
  free_device_memory(&ptr);
  EXPECT_EQ(ptr, nullptr);
}

TEST(CUDA_MANAGEMENT_TESTS, CUDA_RUNNER_TEST){
  int function_flag = 0;
  dicomNodeError_t error_flag = dicomNodeError_t::SUCCESS;

  auto error_function = [&]([[maybe_unused]] const dicomNodeError_t input) noexcept {
    error_flag = input;
  };

  DicomNodeRunner runner{error_function};
  runner | [&]() noexcept {
      function_flag = 1;
      return cudaSuccess;
    } | []() noexcept {
      return cudaErrorInitializationError;
    } | [&]() noexcept {
      function_flag = 2;
      return cudaSuccess;
    };

  EXPECT_TRUE(function_flag == 1);
  EXPECT_TRUE(extract_cuda_error(error_flag) == cudaErrorInitializationError);
  EXPECT_TRUE(extract_cuda_error(runner.error()) == cudaErrorInitializationError);
}

