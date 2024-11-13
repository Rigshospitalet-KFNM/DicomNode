#pragma once

#include"core/core.cuh"

namespace {
  template<typename T>
  __global__ void kernel_interpolation_linear(
    const Texture* src_image,
    const Space<3>* dst_space,
    T* destination_data
  ){
    const Texture& texture = *src_image;
    const Space<3>& destination_space = *dst_space;
    const uint64_t gid = blockDim.x * blockIdx.x + threadIdx.x;

    size_t destination_elements = 1;
    for (const int& dim : destination_space.domain){
      destination_elements *= dim;
    }

    if(gid < destination_elements){
      Index<3> index = destination_space.index(gid);
      Point<3> point = destination_space.at_index(index);
      destination_data[gid] = texture(point);
    }
  }
}

template<typename T>
dicomNodeError_t gpu_interpolation_linear(
  const Texture* device_image,
  const Space<3>& host_destination_space,
  T* device_out_data
){

  Space<3>* device_space = nullptr;

  int cudaDevice;
  cudaDeviceProp prop;

  size_t block_count;
  constexpr size_t threads = 1024;

  DicomNodeRunner runner{[&](dicomNodeError_t _){
    free_device_memory(&device_space);
  }};

  runner
    | [&](){ return cudaGetDevice(&cudaDevice);}
    | [&](){ return cudaGetDeviceProperties(&prop, cudaDevice);}
    | [&](){ return cudaMalloc(&device_space, sizeof(Space<3>));}
    | [&](){
      size_t elements = 1;
      for (const int& dim : host_destination_space.domain){
        elements *= dim;
      }

      //size_t max_threads = prop.multiProcessorCount
      //                   * prop.maxThreadsPerMultiProcessor;
      block_count = elements;

      kernel_interpolation_linear<T><<<block_count, threads>>>(
        device_image,
        device_space,
        device_out_data
      );
      return cudaGetLastError();
    }
    | [&](){
      free_device_memory(&device_space);
      return dicomNodeError_t::SUCCESS;
    };

  return runner.error();
}
