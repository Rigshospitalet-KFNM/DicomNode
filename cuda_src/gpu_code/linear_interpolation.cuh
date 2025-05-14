#pragma once

#include"core/core.cuh"

namespace {
  template<typename T>
  __global__ void kernel_interpolation_linear(
    const Texture<3, T>* src_image,
    const Space<3>* dst_space,
    T* destination_data
  ){
    const Texture<3, T>& texture = *src_image;
    const Space<3>& destination_space = *dst_space;

    size_t destination_elements = destination_space.extent[0] *
                                  destination_space.extent[1] *
                                  destination_space.extent[2];

    const uint64_t gid = blockDim.x * blockIdx.x + threadIdx.x;

    if(gid < destination_elements){
      const Index<3> index = destination_space.index(gid);
      const Point<3> point = destination_space.at_index(index);
      destination_data[gid] = texture(point);
    }
  }
}

template<typename T>
dicomNodeError_t gpu_interpolation_linear(
  const Texture<3, T>* device_texture,
  const Space<3>& host_destination_space,
  T* device_out_data
){
  Space<3>* device_space = nullptr;

  int cudaDevice;
  cudaDeviceProp prop; // This is to calculate the "wave size" for the device

  size_t block_count;
  constexpr size_t threads = 1024;

  DicomNodeRunner runner{[&](dicomNodeError_t _){
    free_device_memory(&device_space);
  }};

  runner
    | [&](){ return cudaGetDevice(&cudaDevice);}
    | [&](){ return cudaGetDeviceProperties(&prop, cudaDevice);}
    | [&](){ return cudaMalloc(&device_space, sizeof(Space<3>));}
    | [&](){ return cudaMemcpy(device_space, &host_destination_space, sizeof(Space<3>), cudaMemcpyDefault);}
    | [&](){
      const size_t elements = host_destination_space.extent[0] *
                              host_destination_space.extent[1] *
                              host_destination_space.extent[2];

      //size_t max_threads = prop.multiProcessorCount
      //                   * prop.maxThreadsPerMultiProcessor;
      block_count = elements % threads ? (elements / threads ) +1 : elements / threads;
      kernel_interpolation_linear<T><<<block_count, threads>>>(
        device_texture,
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
