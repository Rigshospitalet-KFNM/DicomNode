#pragma once

#include"core/core.cuh"

namespace {
  template<typename T>
  __global__ void kernel_interpolation_linear(
    const Image<3, T>* src_image,
    const Space<3> dst_space,
    T* dst_data
  ){
    const T* src_data = src_image->data;

  }
}



template<typename T>
dicomNodeError_t gpu_interpolation_linear(
  Image<3, T>* device_image,
  Space<3>& host_new_space,
  T* device_out_data
){

  Space<3>* device_space = nullptr;

  DicomNodeRunner runner{[&](dicomNodeError_t _){
    free_device_memory(&device_space)
  }};

  size_t block_count = 1;

  runner
    | [&](){ return cudaMalloc(&device_space, sizeof(Space<3>));}
    | [&](){
      kernel_interpolation_linear<T><<<block_count, 1024>>>(
        device_image,
        device_space,
        device_out_data
      );
      return cudaGetLastError();
    }
    | [&](){
      free_device_memory(&device_space);
      return dicomNodeError_t::SUCCESS;
    }



  return dicomNodeError_t::SUCCESS;
}
