#pragma once

# include "../gpu_code/core/core.cuh"



template <typename T, u64 length>
constexpr cuda::std::array<T, length> create_blank_array() {
  return cuda::std::array<T, length>{};
}

struct Bound {
  u32 lower = 0;
  u32 upper = 0;

  bool contains(const i32 i) const noexcept {
    return cuda::std::cmp_less_equal(lower, i) && cuda::std::cmp_less_equal(i, upper);
  }
};

template<typename T, u8 dimensionality>
struct ArrayDataBlock {
  T value;
  cuda::std::array<Bound, dimensionality> bounds; // X bounds, Y bounds, Z Bounds, ...

  bool contains(std::array<i32, dimensionality>& index) const {
    bool ret = true;

    for (int d = 0; d < dimensionality; ++d) {
      ret &= bounds.contains(index[d]);
    }

    return ret;
  }
};

template<typename T, u64 len>
void apply_bound(cuda::std::array<T, len>& array, const Extent<3>& extent, const ArrayDataBlock<T, 3>& block) {
  assert(extent.elements() == len);

  for (int z = block.bounds[2].lower; z < block.bounds[2].upper; ++z) {
    for (int y = block.bounds[1].lower; y < block.bounds[1].upper; ++y) {
      for (int x = block.bounds[0].lower; x < block.bounds[0].upper; ++x) {
        Index<3> index(x, y, z);

        array[extent.flat_index(index)] = block.value;
      }
    }
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