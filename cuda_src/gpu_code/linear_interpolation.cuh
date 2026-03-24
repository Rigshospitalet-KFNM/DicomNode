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

    const uint64_t gid = get_gid();

    if(gid < destination_elements){
      const Index<3> index = destination_space.index(gid);
      const Point<3> point = destination_space.at_index(index);
      destination_data[gid] = texture(point);
    }
  }


  template<typename T>
  __global__ void kernel_interpolation_linear(
    const Image<3, T>* src_image,
    const Space<3>* dst_space,
    T* destination_data
  ) {

    const size_t destination_elements = dst_space->elements();
    const u64 gid = get_gid();

    if (gid < destination_elements) {
      const Index<3> gid_index = dst_space->index(gid);
      const Point<3> point_in_destination_space = dst_space->at_index(gid_index);
      const Point<3> point_in_source_space = src_image->space.interpolate_point(point_in_destination_space);
      destination_data[gid] = src_image->volume.interpolate_at_index_point(point_in_source_space);
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

  constexpr size_t threads = 1024;

  DicomNodeRunner runner{[&](dicomNodeError_t _){
    free_device_memory(&device_space);
  }};

  runner
    | [&](){ return cudaMalloc(&device_space, sizeof(Space<3>));}
    | [&](){ return cudaMemcpy(device_space, &host_destination_space, sizeof(Space<3>), cudaMemcpyDefault);}
    | [&](){
      const size_t elements = host_destination_space.extent[0] *
                              host_destination_space.extent[1] *
                              host_destination_space.extent[2];

      const size_t block_count = elements % threads ? (elements / threads ) + 1 : elements / threads;
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

template<typename T>
dicomNodeError_t gpu_interpolation_linear(
  const Image<3, T>& host_image,
  const Space<3>& host_destination_space,
  T* device_out_data
){

  constexpr u64 threads = 1024;

  Space<3>* device_space = nullptr;
  Image<3, T>* device_image = nullptr;

  DicomNodeRunner runner{[&](const dicomNodeError_t error){
    print_error(error);
    free_device_memory(&device_space, &device_image);
  }};

  runner
    | [&](){ return cudaMalloc(&device_image, sizeof(Image<3, T>)); }
    | [&](){ return cudaMalloc(&device_space, sizeof(Space<3>));}
    | [&](){ return cudaMemcpy(device_image, &host_image, sizeof(Image<3, T>), cudaMemcpyDefault);}
    | [&](){ return cudaMemcpy(device_space, &host_destination_space, sizeof(Space<3>), cudaMemcpyDefault);}
    | [&](){
      const size_t elements = host_destination_space.extent[0] *
                              host_destination_space.extent[1] *
                              host_destination_space.extent[2];

      const u64 block_count = elements % threads ? (elements / threads ) + 1 : elements / threads;
      kernel_interpolation_linear<T><<<block_count, threads>>>(
        device_image,
        device_space,
        device_out_data
      );
      return cudaGetLastError();
    } | [&](){
      free_device_memory(&device_space, &device_image);
      return dicomNodeError_t::SUCCESS;
    };

  return runner.error();

}
