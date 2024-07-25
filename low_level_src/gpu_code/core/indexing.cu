#pragma once

// Standard library
#include<stdint.h>
#include<cuda/std/optional>

template<uint8_t>
struct Space;

template<uint8_t DIMENSIONS>
struct Index {
  static_assert(DIMENSIONS > 0);
  // Negative number would indicate and out of image index
  int32_t coordinates[DIMENSIONS];

  template<typename... Args>
  __device__ __host__ Index(const Args... args){
    static_assert(sizeof...(args) == DIMENSIONS);

    int32_t temp[] = {static_cast<int32_t>(args)...};
    #pragma unroll
    for(uint8_t dim = 0; dim < DIMENSIONS; dim++){
      coordinates[dim] = temp[dim];
    }
  };

  __device__ __host__ Index(const uint64_t flat_index, const Space<DIMENSIONS> space){
    uint64_t dimension_temp = 1;

    #pragma unroll
    for(uint8_t dim = 0; dim < DIMENSIONS; dim++){
      coordinates[dim] = (flat_index % (dimension_temp * space[dim]))
        / dimension_temp;
      dimension_temp *= space[dim];
    }
  }

  __device__ __host__ Index(const int32_t* temp){
    #pragma unroll
    for(uint8_t dim = 0; dim < DIMENSIONS; dim++){
      coordinates[dim] = temp[dim];
    }
  }

  __device__ __host__  const int32_t& operator[](const uint8_t idx){
    // you can't put a static assert in here :(
    return coordinates[idx];
  }

  __device__ __host__ int32_t operator[](const uint8_t idx) const {
    return coordinates[idx];
  }

  __device__ __host__ const int32_t& x() const {
    return coordinates[0];
  }

  __device__ __host__ const int32_t& y() const {
    static_assert(DIMENSIONS > 1);
    return coordinates[1];
  }

  __device__ __host__ const int32_t& z() const {
    static_assert(DIMENSIONS > 2);
    return coordinates[2];
  }
};

/**
 * @brief Struct containing the size information of an image
 *
 */
template<uint8_t DIMENSIONS>
struct Space {
  static_assert(DIMENSIONS > 0);
  uint32_t sizes[DIMENSIONS];

  template<typename... Args>
  __device__ __host__ Space(Args... args){
    static_assert(sizeof...(args) == DIMENSIONS);

    uint32_t temp[] = {static_cast<uint32_t>(args)...};
    #pragma unroll
    for(uint8_t dim = 0; dim < DIMENSIONS; dim++){
      sizes[dim] = temp[dim];
    }
  };

  __device__ __host__ const uint32_t& operator[](const int i) const {
    return sizes[i];
  }

  __device__ __host__ inline bool contains(const Index<DIMENSIONS> index) const{
    bool in = true;
    #pragma unroll
    for(uint8_t i = 0; i < DIMENSIONS; i++){
      in &= 0 <= index[i] && index[i] < sizes[i];
    }

    return in;
  }

  __device__ __host__ cuda::std::optional<uint64_t> flat_index(const Index<DIMENSIONS> index) const {
    if(!contains(index)){
      return {};
    }

    uint64_t return_val = 0;
    uint64_t dimension_temp = 1;

    #pragma unroll
    for(uint8_t i = 0; i < DIMENSIONS; i++){
      return_val += index[i] * dimension_temp;
      dimension_temp *= sizes[i];
    }

    return return_val;
  }

  template<typename... Args>
  __device__ __host__ bool contains(const Args... args) const {
    static_assert(sizeof...(args) == DIMENSIONS);
    int32_t temp[] = {static_cast<int32_t>(args)...};

    bool in = true;
    #pragma unroll
    for(uint8_t i = 0; i < DIMENSIONS; i++){
      in &= 0 <= temp[i] && temp[i] < sizes[i];
    }

    return in;
  }

  template<typename... Args>
  __device__ __host__ cuda::std::optional<uint64_t> flat_index(const Args... args) const {
    static_assert(sizeof...(args) == DIMENSIONS);
    Index index = Index<DIMENSIONS>(args...);

    return flat_index(index);
  }

  __device__ __host__ const uint32_t& x() const {
    return sizes[0];
  }

  __device__ __host__ const uint32_t& y() const {
    static_assert(DIMENSIONS > 1);
    return sizes[1];
  }

  __device__ __host__ const uint32_t& z() const {
    static_assert(DIMENSIONS > 2);
    return sizes[2];
  }

  __device__ __host__ Index<DIMENSIONS> from_flat_index(uint64_t flat_index) const {
    int32_t coordinates[DIMENSIONS];
    uint64_t dimension_temp = 1;

    #pragma unroll
    for(uint8_t dim = 0; dim < DIMENSIONS; dim++){
      coordinates[dim] = (flat_index % (dimension_temp * sizes[dim]))
        / dimension_temp;
      dimension_temp *= sizes[dim];
    }

    return Index<DIMENSIONS>(coordinates);
  }
};

template<uint8_t DIMENSIONS>
struct ImagePadding {
  uint8_t m[DIMENSIONS]; // minus
  uint8_t p[DIMENSIONS]; // plus
};

template<typename T, ImagePadding padding>
class Volume {
  volatile T* image;
  Space<3> dimensions;
  T default_value;

  public:
    __device__ Volume(T* source_image,
                      const Space<3>& source_dimentions,
                      volatile T* destination_image,
                      const T default_value_) : image(destination_image), dimensions(
                        padding.m[0] + blockDim.x + padding.p[0],
                        padding.m[1] + blockDim.y + padding.p[1],
                        padding.m[2] + blockDim.z + padding.p[2]
                      ), default_value(default_value_){

    const uint16_t size = dimensions.x() * dimensions.y() * dimensions.z();
    const uint16_t threads = blockDim.x * blockDim.y * blockDim.z;
    const uint16_t tid = blockDim.x * blockDim.y * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x;
    const uint32_t starting_index_x = blockDim.x * blockIdx.x - padding.m[0];
    const uint32_t starting_index_y = blockDim.y * blockIdx.y - padding.m[1];
    const uint32_t starting_index_z = blockDim.z * blockIdx.z - padding.m[2];

    for(int32_t li = tid; li < size; li += threads){
      const int16_t li_x = li % blockDim.x;
      const int16_t li_y = (li / blockDim.x) % blockDim.y;
      const int16_t li_z = li / (blockDim.x * blockDim.y);

      const int32_t gi_x = starting_index_x + li_x;
      const int32_t gi_y = starting_index_y + li_y;
      const int32_t gi_z = starting_index_z + li_z;
      const cuda::std::optional<int64_t> flat_index = dimensions.flat_index(gi_x, gi_y, gi_z);
      image[li] = flat_index.has_value() ? source_image[flat_index.value()] : default_value;
    }
  }

  __device__ T index(const int32_t x, const int32_t y, const int32_t z){
    const int32_t idx = x + padding.m[0];
    const int32_t idy = y + padding.m[1];
    const int32_t idz = z + padding.m[2];
    const cuda::std::optional<int64_t> flat_index = dimensions.flat_index(idx, idy, idz).value();
    return flat_index.has_value() ? image[flat_index.value()] : default_value;
  }
};

template<typename T, ImagePadding<1> padding>
class Line {
  volatile T* image;
  Space<1> dimensions;
  T default_value;

  public:
    __device__ Line(T* source_image,
                    const Space<1>& source_dimentions,
                    volatile T* destination_image,
                    const T default_value_) : image(destination_image), dimensions(
                        padding.m[0] + blockDim.x + padding.p[0]
                      ), default_value(default_value_){

    const uint16_t size = dimensions.x();
    const uint16_t threads = blockDim.x;
    const uint16_t tid = threadIdx.x;
    const uint32_t starting_index_x = blockDim.x * blockIdx.x - padding.m[0];

    for(int32_t li = tid; li < size; li += threads){
      const int32_t li_x = li;

      const int32_t gi_x = starting_index_x + li_x;
      const cuda::std::optional<int64_t> flat_index = dimensions.flat_index(gi_x);
      image[li] = flat_index.has_value() ? source_image[flat_index.value()] : default_value;
    }
  }

  __device__ __host__ T index(const int32_t x){
    const int32_t idx = x + padding.m[0];
    const cuda::std::optional<int64_t> flat_index = dimensions.flat_index(idx).value();
    return flat_index.has_value() ? image[flat_index.value()] : default_value;
  }
};

template<typename T, ImagePadding padding>
class Plane {
  T* image;
  Space<2> dimensions;
  T default_value;
  public:
    __device__ Plane(T* source_image,
                     const Space<2>& source_dimentions,
                     T* destination_image,
                     const T default_value_) : image(destination_image), dimensions(
                        padding.m[0] + blockDim.x + padding.p[0],
                        padding.m[1] + blockDim.y + padding.p[1]
                      ), default_value(default_value_) {

    const uint16_t size = dimensions.x() * dimensions.y();
    const uint16_t threads = blockDim.x * blockDim.y;
    const uint16_t tid = blockDim.x * threadIdx.y + threadIdx.x;
    const uint32_t starting_index_x = blockDim.x * blockIdx.x - padding.m[0];
    const uint32_t starting_index_y = blockDim.y * blockIdx.y - padding.m[1];

    for(int32_t li = tid; li < size; li += threads){
      const int16_t li_x = li % blockDim.x;
      const int16_t li_y = li / blockDim.x;

      const int32_t gi_x = starting_index_x + li_x;
      const int32_t gi_y = starting_index_y + li_y;
      const cuda::std::optional<int64_t> flat_index = dimensions.flat_index(gi_x, gi_y);
      image[li] = flat_index.has_value() ? source_image[flat_index.value()] : default_value;
    }
  }

  __device__ __host__ T index(const int32_t x, const int32_t y){
    const int32_t idx = x + padding.m[0];
    const int32_t idy = y + padding.m[1];
    const cuda::std::optional<int64_t> flat_index = dimensions.flat_index(idx, idy).value();
    return flat_index.has_value() ? image[flat_index.value()] : default_value;
  }
};
