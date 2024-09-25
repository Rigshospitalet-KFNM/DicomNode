#pragma once

#include<stdint.h>
#include"concepts.cuh"

template<uint8_t CHUCK, typename OP, typename T_IN, typename T_OUT, typename... Args>
  requires MappingBinaryOperator<OP, T_IN, T_OUT, Args...>
__device__ inline void map_into_shared_memory(
    volatile T_OUT* dst_shared,
    T_IN* src_global,
    size_t number_of_elements,
    size_t offset,
    Args... args
  ){
  #pragma unroll
  for(uint8_t i=0; i < CHUCK; i++){
    const uint16_t local_index = threadIdx.x + blockDim.x * i;
    const size_t global_index = local_index + offset;
    T_OUT element = OP::identity();
    if(global_index < number_of_elements){
      element = OP::map_to(src_global[global_index], global_index, args...);
    }
    dst_shared[local_index] = element;
  }

  __syncthreads();
}

template<uint8_t CHUCK, typename T>
__device__ inline void map_into_shared_memory(
    volatile T* dst_shared,
    T* src_global,
    size_t number_of_elements,
    size_t offset
  ){
  #pragma unroll
  for(uint8_t i=0; i < CHUCK; i++){
    const uint16_t local_index = threadIdx.x + blockDim.x * i;
    const size_t global_index = local_index + offset;
    T element = 0;
    if(global_index < number_of_elements){
      element = src_global[global_index];
    }
    dst_shared[local_index] = element;
  }

  __syncthreads();
}

template<typename T, uint8_t CHUCK = 1>
__device__ inline void copy_from_shared_memory(volatile T* dst_global, volatile T* src_shared, size_t number_of_elements, size_t offset=0){
  #pragma unroll
  for(uint8_t i=0; i < CHUCK; i++){
    const uint32_t local_index = threadIdx.x + blockDim.x * i;
    const size_t global_index = local_index + offset;

    if(global_index < number_of_elements){
      dst_global[global_index] = src_shared[local_index];
    }
  }

  __syncthreads();
}