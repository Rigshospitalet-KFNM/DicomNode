#pragma once

# include "../gpu_code/core/core.cuh"



template <typename T, u64 length>
constexpr cuda::std::array<T, length> create_blank_array() {
  return cuda::std::array<T, length>{};
}

struct bound {
  u32 lower = 0;
  u32 upper = 0;

  bool contains(const i32 i) const noexcept {
    return cuda::std::cmp_less_equal(lower, i) && cuda::std::cmp_less_equal(i, upper);
  }
};

template<typename T, u8 dimensionality>
struct ArrayDataBlock {
  T value;
  cuda::std::array<bound, dimensionality> bounds; // X bounds, Y bounds, Z Bounds, ...

  bool contains(std::array<i32, dimensionality>& index) const {
    bool ret = true;

    for (int d = 0; d < dimensionality; ++d) {
      ret &= bounds.contains(index[d]);
    }

    return ret;
  }
};

template<typename T, u64 len, u8 dimensionality>
constexpr void apply_bound(cuda::std::array<T, len>& array, const Extent<dimensionality>& extent, const ArrayDataBlock<T, dimensionality>& block) {
  std::array<i32, dimensionality> pivot_arr{};

  assert(extent.elements() == len);

  for (int d = 0; d < dimensionality; d++) {
    pivot_arr[d] = block.bounds[d].lower;
  }

  for (int d = 0; d < dimensionality; d++) {

  }




}


template<typename T>
class DeviceArray {
  T* device_ptr = nullptr;
  size_t elements;

public:
  explicit DeviceArray(const size_t count) noexcept  : elements(count) {
    cudaMalloc(&device_ptr, sizeof(T) * elements);
  }

  ~DeviceArray() {
    cudaFree(device_ptr);
  }

  T* get(){ return device_ptr; }
  size_t size() { return elements * sizeof(T); }

  DeviceArray(const DeviceArray&) = delete;
  DeviceArray& operator=(const DeviceArray&) = delete;
  DeviceArray(DeviceArray&& other): device_ptr(other.device_ptr), elements(other.elements) {
    other.device_ptr = nullptr;
    other.elements = 0;
  }

  DeviceArray& operator=(DeviceArray&& other) {
    if (device_ptr) {
      cudaFree(device_ptr);
    }
    device_ptr = other.device_ptr;
    elements = other.elements;

    other.device_ptr = nullptr;
    other.elements = 0;
  }
};