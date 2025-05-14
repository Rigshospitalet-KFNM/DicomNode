#pragma once

#include<stdint.h>

#include"core/core.cuh"

namespace {

template<typename OP, typename T_IN, typename T_OUT, typename... Args>
  requires Mapping<OP, T_IN, T_OUT, Args...>
__global__ void map_kernel(T_IN* in, T_OUT* out, size_t data_size, Args... args){
  const uint64_t global_thread_index = blockDim.x * blockIdx.x + threadIdx.x;

  if(global_thread_index < data_size){
    out[global_thread_index] = OP::map_to(in[global_thread_index], global_thread_index, args...);
  }
}

} // end of anonymous namespace

template<typename OP, typename T_IN, typename T_OUT, typename... Args>
  requires Mapping<OP, T_IN, T_OUT, Args...>
cudaError_t map(T_IN* data_in, T_OUT* data_out, size_t data_size, Args... args){
  T_IN* device_data_in = nullptr;
  T_OUT* device_data_out = nullptr;

  auto error_function = [&](cudaError_t error){
    free_device_memory(&device_data_in, &device_data_out);
  };

  const dim3 grid = get_grid<1>(data_size, MAP_BLOCK_SIZE);

  CudaRunner runner{error_function};

  // So here there's a mirco optimization of just allocating once and indexing
  // I should write some benchmarks to measure just how much it effects stuff
  runner
    | [&](){ return cudaMalloc(&device_data_in, sizeof(T_IN) * data_size); }
    | [&](){ return cudaMalloc(&device_data_out, sizeof(T_OUT) * data_size); }
    | [&](){ return cudaMemcpy(device_data_in, data_in, sizeof(T_IN) * data_size, cudaMemcpyDefault); }
    | [&](){
      map_kernel<OP, T_IN, T_OUT, Args...><<<grid, MAP_BLOCK_SIZE>>>(
        device_data_in, device_data_out, data_size, args...
      );
      return cudaGetLastError(); }
    | [&](){ return cudaMemcpy(data_out, device_data_out, sizeof(T_OUT) * data_size, cudaMemcpyDefault);}
    | [&](){
      free_device_memory(&device_data_in, &device_data_out);
      return cudaSuccess;
    };

  return runner.error();
}
