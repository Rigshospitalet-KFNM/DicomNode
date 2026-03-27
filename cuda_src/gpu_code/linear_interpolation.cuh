#pragma once

#include"core/core.cuh"

namespace INTERPOLATION {
  template<typename T>
  __global__ void kernel_interpolation_linear(
    const Texture<3, T>* src_image,
    const Space<3>* dst_space,
    T* destination_data
  ){
    const Texture<3, T>& texture = *src_image;
    const Space<3>& destination_space = *dst_space;

    const uint64_t gid = get_gid();

    if(gid < destination_space.elements()){
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

  template<typename T>
  __global__ void kernel_interpolation_linear_blocked(
    const Image<3, T>* src_image,
    const Space<3>* dst_space,
    T* destination_data
  ) {

    const size_t destination_elements = dst_space->elements();

    const Index<3> global_index = get_gidx();

    const FlatIndex gid = dst_space->extent.flat_index(global_index);

    if (gid.has_value()) {
      const Point<3> point_in_destination_space = dst_space->at_index(global_index);
      const Point<3> point_in_source_space = src_image->space.interpolate_point(point_in_destination_space);
      destination_data[*gid] = src_image->volume.interpolate_at_index_point(point_in_source_space);
    }
  }

  template<typename T>
  __global__ void kernel_interpolation_linear_shared(
    const Image<3, T>* src_image,
    const Space<3>* dst_space,
    T* destination_data
  ) {
    // Just for my sanity
    const Image<3,T>& source_image = *src_image;
    const Space<3>& destination_space = *dst_space;

    constexpr Extent<3> shared_extent{THREAD_BLOCK_3D.z + 1, THREAD_BLOCK_3D.y + 1, THREAD_BLOCK_3D.x + 1 };
    __shared__ T shared_image_ptr[(THREAD_BLOCK_3D.x + 1)*(THREAD_BLOCK_3D.y + 1)*(THREAD_BLOCK_3D.z + 1)];
    const Index<3> offset_index{
      blockDim.x * blockIdx.x,
      blockDim.y * blockIdx.y,
      blockDim.z * blockIdx.z,
    };


    Image<3, T> shared_image = sub_image<T>(
      source_image,
      shared_image_ptr,
      shared_extent,
      offset_index
    );

    const Index<3> global_index = get_gidx();
    FlatIndex flat_global_index = destination_space.extent.flat_index(global_index);
    if (flat_global_index.has_value()) {
      const Point<3> point_in_destination_space = destination_space.at_index(global_index);
      const Point<3> point_in_source_space = shared_image.space.interpolate_point(point_in_destination_space);
      destination_data[*flat_global_index] = shared_image.volume.interpolate_at_index_point(point_in_source_space);
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

  constexpr size_t THREADS = 1024;

  DicomNodeRunner runner{[&](dicomNodeError_t _){
    free_device_memory(&device_space);
  }};

  runner
    | [&](){ return cudaMalloc(&device_space, sizeof(Space<3>));}
    | [&](){ return cudaMemcpy(device_space, &host_destination_space, sizeof(Space<3>), cudaMemcpyDefault);}
    | [&](){
      const u32 block_count = envelope_length<THREADS>(host_destination_space.elements());
      INTERPOLATION::kernel_interpolation_linear<T><<<block_count, THREADS>>>(
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

  constexpr u64 THREADS = 1024;

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
      const u32 block_count = envelope_length<THREADS>(host_destination_space.elements());
      INTERPOLATION::kernel_interpolation_linear<T><<<block_count, THREADS>>>(
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

template<typename T, auto Kernel>
dicomNodeError_t gpu_interpolation_linear_t(
  const Image<3, T>& host_image,
  const Space<3>& host_destination_space,
  T* device_out_data
) {
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
    const dim3 envelope_grid = get_envelope_grid<THREAD_BLOCK_3D>(host_destination_space.extent);
    Kernel<<<envelope_grid, THREAD_BLOCK_3D>>>(
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
