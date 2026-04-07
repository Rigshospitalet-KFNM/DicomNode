/**
 * @file volume.cuh
 * @Demiguard (you@domain.com)
 * @brief Contains the Volume class
 * @version 0.1
 * @date 2025-02-27
 *
 *
 */
#pragma once

#include<vector>

#include"../concepts.cuh"
#include"extent.cuh"
#include"point.cuh"
#include"index.cuh"
#include"flat_index.cuh"

/** A volume is a memory region, that is constrained by an extent.
 *
 * @tparam DIMENSIONS - The Dimensionality of the volume
 * @tparam T - The Datatype of underlying data
 */
template<uint8_t DIMENSIONS, typename T>
struct Volume {
  T* data = nullptr;
  Extent<DIMENSIONS> m_extent;
  T default_value = 0;

  /* Byte size for the volume
   *
   * @return Number of bytes required to contain the volume in data member
   */
  [[nodiscard]] constexpr __device__ __host__ size_t size() const {
    return m_extent.elements() * sizeof(T);
  }

  constexpr  __device__ __host__  size_t elements() const {
    return m_extent.elements();
  }

  [[nodiscard]] constexpr  __device__ __host__  bool is_allocated() const {
    return data == nullptr;
  }

  dicomNodeError_t set_extent(const std::vector<ssize_t>& dims){
    return m_extent.set_dimensions(dims);
  }

  constexpr __device__ __host__ const Extent<DIMENSIONS>& extent() const {
    return m_extent;
  }

  [[nodiscard]] constexpr __device__ __host__ const T& at(const Index<DIMENSIONS>& index) const {
    const FlatIndex flat_index = m_extent.flat_index(index);

    if(!flat_index.has_value()){
      return default_value;
    }

    return data[flat_index.value()];
  }

  [[nodiscard]] constexpr __device__ __host__ const T& at(const u32& flat_index) const {
    Index<3> idx = m_extent.from_flat_index(flat_index);

    if (!m_extent.contains(idx)) {
      return default_value;
    }

    return data[flat_index];
  }

  [[nodiscard]] constexpr __device__ __host__ const T& at(const u64& flat_index) const {
    Index<3> idx = m_extent.from_flat_index(flat_index);

    if (!m_extent.contains(idx)) {
      return default_value;
    }

    return data[flat_index];
  }

  [[nodiscard]] __device__ __host__ T interpolate_at_index_point(const Point<DIMENSIONS>& p) const {
    T value = 0;

    float alphas[DIMENSIONS];
    Index<DIMENSIONS> index_lower_point;

    for (u8 d = 0; d < DIMENSIONS; d++) {
      alphas[d] = p[d] - floorf(p[d]);
      index_lower_point[d] = floorf(p[d]);
    }

    // A N-dimensional cube has 2^N points
    constexpr static u16 MAX_POINTS = 1 << DIMENSIONS;

    // Coordinate index is the point index of points we need to visit
    for (u8 coordinate_index = 0; coordinate_index < MAX_POINTS; coordinate_index++) {
      float modifier = 1.0f;
      for (u8 d = 0; d < DIMENSIONS; d++ ) {
        modifier *= (coordinate_index >> d) & 1u ? alphas[d] : 1.0f - alphas[d];
      }
      Index<DIMENSIONS> index = index_lower_point + dimensional_offset<DIMENSIONS>(coordinate_index);

      value += modifier * this->at(index);
    }

    return value;
  }
};

template<typename T>
[[nodiscard]] __device__ __host__ Volume<3, T> sub_volume(
    const Volume<3, T>& original_volume,
    T* sub_volume_data_ptr,
    Extent<3> new_extent,
    const Index<3>& offset_index
  ){

  #ifdef __CUDA_ARCH__
  const u64 thread_flat = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
  const u64 thread_stride = blockDim.x * blockDim.y * blockDim.z;

  for(u64 flat_index = thread_flat; flat_index < new_extent.elements(); flat_index += thread_stride){
    const Index<3> local_index = new_extent.from_flat_index(flat_index);
    const Index<3> original_index = offset_index + local_index;

    sub_volume_data_ptr[flat_index] = original_volume.at(original_index);

  }
  __syncthreads();
  #else

  for( i32 z = 0; z < new_extent.z(); z++){
    for( i32 y = 0; y < new_extent.y(); y++){
      for( i32 x = 0; x < new_extent.x(); x++){
        Index<3> local_index{x,y,z};
        Index<3> global_index = local_index + offset_index;
        FlatIndex local_flat_index = new_extent.flat_index(local_index);

        if(local_flat_index.has_value()){
          sub_volume_data_ptr[local_flat_index] = original_volume.at(global_index);
        }
      }
    }
  }

  #endif

  return {
    .data=sub_volume_data_ptr,
    .m_extent=new_extent,
    .default_value=original_volume.default_value
  };
}

//static_assert(CVolume<Volume, 3, float>, "Volume doesn't fulfill volume concepts");

template<u8 DIMENSION, typename T>
struct VOLUME_SUM_OP {
  static __device__ __host__ T map_to(u64 global_index, const Volume<DIMENSION, T>& vol) noexcept {
    return vol.at(global_index);
  }

  static __device__ __host__ T apply(const T& a, const T& b) noexcept {
    return a + b;
  }

  static __device__ __host__ bool equals(const T& a, const T& b) noexcept {
    return a == b;
  }

  static __device__ __host__ T identity() noexcept {
    return 0;
  }

  static __device__ __host__ T remove_volatile(volatile T& v) noexcept {
    T vv = v;
    return vv;
  }
};