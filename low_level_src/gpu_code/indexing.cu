#ifndef DICOMNODE_INDEXING
#define DICOMNODE_INDEXING

// Standard library
#include<stdint.h>
#include<tuple>

struct ImageDimensions {
  uint32_t x,y,z;
};

struct ImagePadding {
  uint32_t x_m,y_m,z_m,x_p,y_p,z_p;
};

struct Index {
  uint32_t index;
  char bit_flags;

  __device__ __host__ Index(const uint32_t x,
                            const uint32_t y,
                            const uint32_t z,
                            const uint32_t x_max,
                            const uint32_t y_max,
                            const uint32_t z_max){
    index = x
          + y * x_max
          + z * x_max + y_max;

    bit_flags = 0;
    bit_flags |= x < x_max && y < y_max && z < z_max;
    bit_flags |= (x + 1 < x_max) << 1;
    bit_flags |= (y + 1 < y_max) << 2;
    bit_flags |= (z + 1 < z_max) << 3;
  };
  ~Index(){}

  __device__ __host__ const bool in_range() const {
    return bit_flags & 1;
  }
  __device__ __host__ const bool incrementable_x() const {
    return bit_flags & 2;
  }
  __device__ __host__ const bool incrementable_y() const {
    return bit_flags & 4;
  }
  __device__ __host__ const bool incrementable_z() const {
    return bit_flags & 8;
  }
};

template<typename T, ImagePadding padding = ImagePadding{0,0,0,0,0,0}>
class CubicSpace {
  T* image;
  ImageDimensions dimensions;
  T default_value;

  public:
    __device__ CubicSpace(T* source_image,
                          const ImageDimensions& source_dimentions,
                          T* destination_image,
                          T default_value_,
                          ){
    image = destination_image;
    dimensions = {
      .x = padding.x_m + blockDim.x + padding.x_p,
      .y = padding.y_m + blockDim.y + padding.y_p,
      .z = padding.z_m + blockDim.z + padding.z_p
    };
    default_value = default_value_;

    const uint16_t threads = blockDim.x * blockDim.y * blockDim.z;
    const uint32_t flatIndex =(gridDim.y * blockDim.y * gridDim.x * blockDim.x) * (blockIdx.y * blockDim.y + ThreadIdx.y) // z index
      + gridDim.x * blockDim.x * (blockIdx.y * blockDim.y + ThreadIdx.y) // y index
      + blockDim.x * blockIdx.x + threadIdx.x; // x index

  }
};

#endif