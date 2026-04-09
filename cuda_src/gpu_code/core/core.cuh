#pragma once

#include<cuda/std/array>

#include"constants.cuh"
#include"declarations.cuh"
#include"error.cuh"
#include"memory.cuh"
#include"concepts.cuh"
#include"cuda_management.cuh"
#include"data_structures/data_structures.cuh"
#include"grid.cuh"
#include"lin_alg.cuh"

template<u64 length, typename F>
constexpr __host__ auto initialize_array(F&& init) {
  using T = std::invoke_result_t<F, u64>;
  return [&]<u64... Is>(cuda::std::index_sequence<Is...>) {
    return cuda::std::array<T, length>{init(Is)...};
  }(cuda::std::make_index_sequence<length>{});
}

enum class DIMENSION {
  X = 0,
  Y = 1,
  Z = 2,
};
