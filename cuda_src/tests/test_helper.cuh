#pragma once

# include "../gpu_code/core/core.cuh"



template <typename T, u64 length>
constexpr cuda::std::array<T, length> create_blank_array() {
  return cuda::std::array<T, length>{};
}

/**
 * Inclusive bound for indexes
 */
struct Bound {
  u32 lower = 0;
  u32 upper = 0;

  [[nodiscard]] constexpr bool contains(const i32 i) const noexcept {
    return cuda::std::cmp_less_equal(lower, i) && cuda::std::cmp_less_equal(i, upper);
  }

  struct iterator {
    u32 value;
    constexpr u32 operator*() const noexcept {
      return value;
    }
    constexpr iterator& operator++() noexcept {
      ++value; return *this;
    }
    constexpr bool operator==(const iterator& other) const noexcept {
      return value == other.value;
    }
  };

  [[nodiscard]] constexpr iterator begin() const noexcept {
    return iterator{lower};
  }
  [[nodiscard]] constexpr iterator end() const noexcept {
    return iterator{upper + 1};
  }
};

template<typename T, u8 dimensionality>
struct ArrayDataBlock {
  cuda::std::array<Bound, dimensionality> bounds; // X bounds, Y bounds, Z Bounds, ...
  T value;

  [[nodiscard]] constexpr bool contains(std::array<i32, dimensionality>& index) const noexcept {
    for (int d = 0; d < dimensionality; ++d) {
      if (!bounds[d].contains(index[d])) {
        return false;
      }
    }

    return true;
  }

  [[nodiscard]] constexpr bool contains(const Index<dimensionality>& index) const noexcept {
    for (i32 d = 0; d < dimensionality; ++d) {
      if (!bounds[d].contains(index[d])) { return false; }
    }

    return true;
  }
};

template<typename T>
ArrayDataBlock(T) -> ArrayDataBlock<T, 3>;

ArrayDataBlock() -> ArrayDataBlock<f32, 3>;

template<typename T, u64 len>
constexpr void apply_bound(
  cuda::std::array<T, len>& array,
  const Extent<3>& extent,
  const ArrayDataBlock<T, 3>& block
) {
  assert(extent.elements() == len);

  for (const int z : block.bounds[2]) {
    for (const int y : block.bounds[1]) {
      for (const int x : block.bounds[0]) {
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

  [[nodiscard]] T* get() noexcept { return device_ptr; }
  [[nodiscard]] size_t size() const noexcept { return elements * sizeof(T); }

  DeviceArray(const DeviceArray&) = delete;
  DeviceArray& operator=(const DeviceArray&) = delete;
  DeviceArray(DeviceArray&& other) noexcept : device_ptr(other.device_ptr), elements(other.elements) {
    other.device_ptr = nullptr;
    other.elements = 0;
  }

  DeviceArray& operator=(DeviceArray&& other) noexcept {
    if (device_ptr) {
      cudaFree(device_ptr);
    }
    device_ptr = other.device_ptr;
    elements = other.elements;

    other.device_ptr = nullptr;
    other.elements = 0;

    return *this;
  }
};