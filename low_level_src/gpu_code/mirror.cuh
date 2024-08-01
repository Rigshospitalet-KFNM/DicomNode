#pragma once

// Standard library
#include<stdint.h>
#include<functional>
#include<iostream>

// Dicomnode imports
#include"core/core.cuh"

namespace {
template<typename T>
__host__ __device__ T invert_index(const T index, const T size){
  return size - index - 1;
}

template<typename OP, typename T, typename... Args>
  requires Mirrors<OP, T, Args...>
__global__ void mirror_kernel(
    T* data_in, T* data_out, size_t data_size, Args... args
  ){

  const uint64_t global_index = blockDim.x * blockIdx.x + threadIdx.x;

  if(global_index < data_size){
    data_out[global_index] = OP::mirrors(data_in, global_index, args...);
  }

  }
} // end of anonymous namespace

template<typename T>
class Mirror_X {
  public:
    static __host__ __device__ T mirrors(const T* data_in,
                                         const uint64_t flat_index,
                                         const Space<3> space
                                        ) {
      const Index<3> index = space.from_flat_index(flat_index);

      const Index<3> inverted_index(
        invert_index(index.x(), (int32_t)space.x()),
        index.y(),
        index.z()
      );

      const cuda::std::optional<uint64_t> o_flat_inverted_index = space.flat_index(inverted_index);

      if(!o_flat_inverted_index.has_value()){
        assert(false);
      }

      const uint64_t flat_inverted_index = o_flat_inverted_index.value();

      return data_in[flat_inverted_index];
    }
};

template<typename T>
class Mirror_Y {
  public:
    static __host__ __device__ T mirrors(const T* data_in,
                                         const uint64_t flat_index,
                                         const Space<3> space
                                        ) {
      const Index<3> index = space.from_flat_index(flat_index);

      const Index<3> inverted_index(
        index.x(),
        invert_index(index.y(), (int32_t)space.y()),
        index.z()
      );

      const cuda::std::optional<uint64_t> o_flat_inverted_index = space.flat_index(inverted_index);

      if(!o_flat_inverted_index.has_value()){
        assert(false);
      }

      const uint64_t flat_inverted_index = o_flat_inverted_index.value();

      return data_in[flat_inverted_index];
    }
};

template<typename T>
class Mirror_Z {
  public:
    static __host__ __device__ T mirrors(const T* data_in,
                                         const uint64_t flat_index,
                                         const Space<3> space
                                        ) {
      const Index<3> index = space.from_flat_index(flat_index);

      const Index<3> inverted_index(
        index.x(),
        index.y(),
        invert_index(index.z(), (int32_t)space.z())
      );

      const cuda::std::optional<uint64_t> o_flat_inverted_index = space.flat_index(inverted_index);

      if(!o_flat_inverted_index.has_value()){
        assert(false);
      }

      const uint64_t flat_inverted_index = o_flat_inverted_index.value();

      return data_in[flat_inverted_index];
    }
};


template<typename T>
class Mirror_XY {
  public:
    static __host__ __device__ T mirrors(const T* data_in,
                                         const uint64_t flat_index,
                                         const Space<3> space
                                        ) {
      const Index<3> index = space.from_flat_index(flat_index);

      const Index<3> inverted_index(
        invert_index(index.x(), (int32_t)space.x()),
        invert_index(index.y(), (int32_t)space.y()),
        index.z()
      );

      const cuda::std::optional<uint64_t> o_flat_inverted_index = space.flat_index(inverted_index);

      if(!o_flat_inverted_index.has_value()){
        assert(false);
      }

      const uint64_t flat_inverted_index = o_flat_inverted_index.value();

      return data_in[flat_inverted_index];
    }
};

template<typename T>
class Mirror_XZ {
  public:
    static __host__ __device__ T mirrors(const T* data_in,
                                         const uint64_t flat_index,
                                         const Space<3> space
                                        ) {
      const Index<3> index = space.from_flat_index(flat_index);

      const Index<3> inverted_index(
        invert_index(index.x(), (int32_t)space.x()),
        index.y(),
        invert_index(index.z(), (int32_t)space.z())
      );

      const cuda::std::optional<uint64_t> o_flat_inverted_index = space.flat_index(inverted_index);

      if(!o_flat_inverted_index.has_value()){
        assert(false);
      }

      const uint64_t flat_inverted_index = o_flat_inverted_index.value();

      return data_in[flat_inverted_index];
    }
};

template<typename T>
class Mirror_YZ {
  public:
    static __host__ __device__ T mirrors(const T* data_in,
                                         const uint64_t flat_index,
                                         const Space<3> space
                                        ) {
      const Index<3> index = space.from_flat_index(flat_index);

      const Index<3> inverted_index(
        index.x(),
        invert_index(index.y(), (int32_t)space.y()),
        invert_index(index.z(), (int32_t)space.z())
      );

      const cuda::std::optional<uint64_t> o_flat_inverted_index = space.flat_index(inverted_index);

      if(!o_flat_inverted_index.has_value()){
        assert(false);
      }

      const uint64_t flat_inverted_index = o_flat_inverted_index.value();

      return data_in[flat_inverted_index];
    }
};


template<typename T>
class Mirror_XYZ {
  public:
    static __host__ __device__ T mirrors(const T* data_in,
                                         const uint64_t flat_index,
                                         const Space<3> space
                                        ) {
      const Index<3> index = space.from_flat_index(flat_index);

      const Index<3> inverted_index(
        invert_index(index.x(), (int32_t)space.x()),
        invert_index(index.y(), (int32_t)space.y()),
        invert_index(index.z(), (int32_t)space.z())
      );

      const cuda::std::optional<uint64_t> o_flat_inverted_index = space.flat_index(inverted_index);

      if(!o_flat_inverted_index.has_value()){
        assert(false);
      }

      const uint64_t flat_inverted_index = o_flat_inverted_index.value();

      return data_in[flat_inverted_index];
    }
};


template<typename OP, typename T>
  requires Mirrors<OP, T, Space<3>>
__host__ cudaError_t mirror(const T* data_in, T* data_out, const Space<3> space){
  T* device_data_in = nullptr;
  T* device_data_out = nullptr;
  const size_t data_size = space.size();

  auto error_function = [&](cudaError_t error){
    free_device_memory(&device_data_in, &device_data_out);
  };

  const dim3 grid = get_grid<1>(data_size, MAP_BLOCK_SIZE);

  CudaRunner runner{error_function};

  // So here there's a mirco optimization of just allocating once and indexing
  // I should write some benchmarks to measure just how much it effects stuff
  runner
    | [&](){ return cudaMalloc(&device_data_in, sizeof(T) * data_size); }
    | [&](){ return cudaMalloc(&device_data_out, sizeof(T) * data_size); }
    | [&](){ return cudaMemcpy(device_data_in, data_in, sizeof(T) * data_size, cudaMemcpyDefault); }
    | [&](){
      mirror_kernel<OP, T, Space<3>><<<grid, MIRROR_BLOCK_SIZE>>>(
        device_data_in, device_data_out, space.size(), space
      );
      return cudaGetLastError(); }
    | [&](){ return cudaMemcpy(data_out, device_data_out, sizeof(T) * data_size, cudaMemcpyDefault);}
    | [&](){
      free_device_memory(&device_data_in, &device_data_out);
      return cudaSuccess;
    };

  return runner.error();
}

template<typename OP, typename T>
  requires Mirrors<OP, T, Space<3>>
__host__ cudaError_t mirror_in_place(T* data_in, const Space<3> space){
  T* device_data_in = nullptr;
  T* device_data_out = nullptr;
  const size_t data_size = space.size();

  auto error_function = [&](cudaError_t error){
    free_device_memory(&device_data_in, &device_data_out);
  };

  const dim3 grid = get_grid<1>(data_size, MAP_BLOCK_SIZE);

  CudaRunner runner{error_function};

  // So here there's a mirco optimization of just allocating once and indexing
  // I should write some benchmarks to measure just how much it effects stuff
  runner
    | [&](){ return cudaMalloc(&device_data_in, sizeof(T) * data_size); }
    | [&](){ return cudaMalloc(&device_data_out, sizeof(T) * data_size); }
    | [&](){ return cudaMemcpy(device_data_in, data_in, sizeof(T) * data_size, cudaMemcpyDefault); }
    | [&](){
      mirror_kernel<OP, T, Space<3>><<<grid, MIRROR_BLOCK_SIZE>>>(
        device_data_in, device_data_out, space.size(), space
      );
      return cudaGetLastError(); }
    | [&](){ return cudaMemcpy(data_in, device_data_out, sizeof(T) * data_size, cudaMemcpyDefault);}
    | [&](){
      free_device_memory(&device_data_in, &device_data_out);
      return cudaSuccess;
    };

  return runner.error();
}
