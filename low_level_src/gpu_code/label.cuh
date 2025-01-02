/**
 * @file label.cuh
 * @author Demiguard (cjen0668@regionh.dk)
 * @brief
 * @version 0.1
 * @date 2024-12-18
 *
 * @copyright Copyright (c) 2024
 *
 */
#include"core/core.cuh"


namespace {
  template<typename T>
  __global__ void label_kernel(T* image, Domain<3> domain){
    // Size is sizeof(T) * block_size (1024 * sizeof(T))
    extern __shared__ char shared_memory[];





  }

}
