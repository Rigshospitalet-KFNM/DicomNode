#include <gtest/gtest.h>

#include"../gpu_code/dicom_node_gpu.cu"

#include<iostream>

TEST(CUDA_MANAGEMENT_TESTS, Free_Device_memory) {
  int* ptr;
  cudaError_t error = cudaMalloc(&ptr, sizeof(int) * 10);
  ASSERT_EQ(error, cudaSuccess);
  free_device_memory(&ptr);
  ASSERT_EQ(ptr, nullptr);
}

TEST(CUDA_MANAGEMENT_TESTS, CUDA_RUNNER_TEST){
  int function_flag = 0;
  cudaError_t error_flag = cudaSuccess;

  auto error_function = [&](cudaError_t input){
    error_flag = input;
  };

  CudaRunner runner{error_function};
  runner | [&](){
      function_flag = 1;
      return cudaSuccess;
    } | [](){
      return cudaErrorInitializationError;
    } | [&](){
      function_flag = 2;
      return cudaSuccess;
    };

  std::cout << (function_flag == 1) << "\n";
  std::cout << (error_flag == cudaErrorInitializationError) << "\n";
  std::cout << (runner.error() == cudaErrorInitializationError) << "\n";
}
