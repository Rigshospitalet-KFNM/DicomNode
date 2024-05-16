#ifndef DICOMNODE_INDEXING
#define DICOMNODE_INDEXING

// Standard library
#include<stdint.h>

/**
 * @brief Struct containing the size information of an image
 *
 */
struct ImageDimensions {
  uint32_t x,y,z;

  /**
   * @brief Checks if the input would index inside of the 1D image
   * Assumes that y = 1 and z = 1!
   * @param x_ the index of the first dimension
   * @return true if in range
   * @return false if not in range
   */
  __device__ __host__ bool contains(int32_t x_) const {
    return 0 <= x_ && x_ < x;
  }

  /**
   * @brief Checks if the input would index inside of the 2D image
   * Assumes that z = 1!
   * @param x_ the index of the first dimension
   * @param y_ the index of the second dimension
   * @return true if in range
   * @return false if not in range
   */
  __device__ __host__ bool contains(int32_t x_, int32_t y_) const {
    return 0 <= x_ && x_ < x && 0 <= y_ &&  y_ < y;
  }

  /**
   * @brief Checks if the input would index inside of the 3D image
   * @param x_ the index of the first dimension
   * @param y_ the index of the second dimension
   * @param z_ the index of the third dimension
   * @return true if in range
   * @return false if not in range
   */
  __device__ __host__ bool contains(int32_t x_, int32_t y_, int32_t z_) const {
    return 0 <= x_ && x_ < x && 0 <= y_ &&  y_ < y &&  0 <= z_ && z_ < z;
  }

  /**
   * @brief Computes the flat index from an 1D index
   *
   * @param idx_x the index of the first dimension
   * @return a Flat index for indexing into a "multidimensional" array
   */
  __device__ __host__ uint32_t index(uint32_t idx_x) const {
    return idx_x;
  }

  /**
   * @brief Computes the flat index from an 2D index
   *
   * @param idx_x the index of the first dimension
   * @param idx_y the index of the second dimension
   * @return a Flat index for indexing into a "multidimensional" array
   */
  __device__ __host__ uint32_t index(uint32_t idx_x, uint32_t idx_y) const {
    return idx_y * x + idx_x;
  }

  /**
   * @brief Computes the flat index from an 3D index
   *
   * @param idx_x the index of the first dimensions
   * @param idx_y the index of the second dimension
   * @param idx_z the index of the third dimension
   * @return a Flat index for indexing into a "multidimensional" array
   */
  __device__ __host__ uint32_t index(uint32_t idx_x, uint32_t idx_y, uint32_t idx_z) const {
    return idx_z * x * y + idx_y * x + idx_x;
  }
};

struct ImagePadding {
  uint32_t x_m,y_m,z_m,x_p,y_p,z_p;
};

struct Index {
  int32_t x;
  int32_t y;
  int32_t z;
};

template<typename T, ImagePadding padding>
class Volume {
  volatile T* image;
  ImageDimensions dimensions;
  T default_value;

  public:
    __device__ Volume(T* source_image,
                      const ImageDimensions& source_dimentions,
                      volatile T* destination_image,
                      const T default_value_){
    image = destination_image;
    dimensions = {
      .x = padding.x_m + blockDim.x + padding.x_p,
      .y = padding.y_m + blockDim.y + padding.y_p,
      .z = padding.z_m + blockDim.z + padding.z_p
    };
    default_value = default_value_;
    const uint16_t size = dimensions.x * dimensions.y * dimensions.z;
    const uint16_t threads = blockDim.x * blockDim.y * blockDim.z;
    const uint16_t tid = blockDim.x * blockDim.y * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x;
    const uint32_t starting_index_x = blockDim.x * blockIdx.x - padding.x_m;
    const uint32_t starting_index_y = blockDim.y * blockIdx.y - padding.y_m;
    const uint32_t starting_index_z = blockDim.z * blockIdx.z - padding.z_m;

    for(int32_t li = tid; li < size; li += threads){
      const int16_t li_x = li % blockDim.x;
      const int16_t li_y = (li / blockDim.x) % blockDim.y;
      const int16_t li_z = li / (blockDim.x * blockDim.y);

      const int32_t gi_x = starting_index_x + li_x;
      const int32_t gi_y = starting_index_y + li_y;
      const int32_t gi_z = starting_index_z + li_z;
      if(source_dimentions.contains(gi_x, gi_y, gi_z)){
        image[li] = source_image[source_dimentions.index(gi_x, gi_y, gi_z)];
      } else {
        image[li] = default_value;
      }
    }
  }

  __device__ T index(const int32_t x, const int32_t y, const int32_t z){
    const int32_t idx = x + padding.x_m;
    const int32_t idy = y + padding.y_m;
    const int32_t idz = z + padding.z_m;
    if(dimensions.contains(idx, idy, idz)){
      return image[dimensions.index(idx, idy, idz)];
    } else {
      return default_value;
    }
  }
};

template<typename T, ImagePadding padding>
class Line {
  volatile T* image;
  ImageDimensions dimensions;
  T default_value;

  public:
    __device__ Line(T* source_image,
                    const ImageDimensions& source_dimentions,
                    volatile T* destination_image,
                    const T default_value_){
    image = destination_image;
    dimensions = {
      .x = padding.x_m + blockDim.x + padding.x_p,
      .y = 1,
      .z = 1
    };
    default_value = default_value_;
    const uint16_t size = dimensions.x;
    const uint16_t threads = blockDim.x;
    const uint16_t tid = threadIdx.x;
    const uint32_t starting_index_x = blockDim.x * blockIdx.x - padding.x_m;

    for(int32_t li = tid; li < size; li += threads){
      const int32_t li_x = li;

      const int32_t gi_x = starting_index_x + li_x;
      if(source_dimentions.contains(gi_x)){
        // This is just here to maintain similairity to the other Spaces
        const uint32_t gi = gi_x;
        image[li_x] = source_image[gi];
      } else {
        image[li_x] = default_value;
      }
    }
  }

  __device__ __host__ T index(const int32_t x){
    const int32_t idx = x + padding.x_m;
    if(dimensions.contains(idx)){
      return image[dimensions.index(idx)];
    } else {
      return default_value;
    }
  }
};

template<typename T, ImagePadding padding>
class Plane {
  T* image;
  ImageDimensions dimensions;
  T default_value;
  public:
    __device__ Plane(T* source_image,
                     const ImageDimensions& source_dimentions,
                     T* destination_image,
                     const T default_value_){
    image = destination_image;
    dimensions = {
      .x = padding.x_m + blockDim.x + padding.x_p,
      .y = padding.y_m + blockDim.y + padding.y_p,
      .z = 1
    };
    default_value = default_value_;
    const uint16_t size = dimensions.x * dimensions.y;
    const uint16_t threads = blockDim.x * blockDim.y;
    const uint16_t tid = blockDim.x * threadIdx.y + threadIdx.x;
    const uint32_t starting_index_x = blockDim.x * blockIdx.x - padding.x_m;
    const uint32_t starting_index_y = blockDim.y * blockIdx.y - padding.y_m;

    for(int32_t li = tid; li < size; li += threads){
      const int16_t li_x = li % blockDim.x;
      const int16_t li_y = li / blockDim.x;

      const int32_t gi_x = starting_index_x + li_x;
      const int32_t gi_y = starting_index_y + li_y;
      if(source_dimentions.contains(gi_x, gi_y)){
        destination_image[li] = source_image[source_dimentions.index(gi_x, gi_y)];
      } else {
        destination_image[li] = default_value;
      }
    }
  }

  __device__ __host__ T index(const int32_t x, const int32_t y){
    const int32_t idx = x + padding.x_m;
    const int32_t idy = y + padding.y_m;
    if(dimensions.contains(idx, idy)){
      return image[dimensions.index(idx, idy)];
    } else {
      return default_value;
    }
  }
};


#endif