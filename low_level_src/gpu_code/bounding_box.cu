#pragma once

#include<exception>

// Thrid party
#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>

// Dicomnode imports
#include"core/core.cu"
#include"map_reduce.cu"

constexpr uint8_t MAX_DIM = 8;


struct BoundingBox_3D {
  uint16_t x_min;
  uint16_t x_max;
  uint16_t y_min;
  uint16_t y_max;
  uint16_t z_min;
  uint16_t z_max;

  __device__ __host__ inline BoundingBox_3D() :
    x_min(UINT16_MAX), x_max(0), y_min(UINT16_MAX), y_max(0), z_min(UINT16_MAX), z_max(0) {}

  __device__ __host__ inline BoundingBox_3D(
    const uint16_t& x_min_,
    const uint16_t& x_max_,
    const uint16_t& y_min_,
    const uint16_t& y_max_,
    const uint16_t& z_min_,
    const uint16_t& z_max_
) : x_min(x_min_), x_max(x_max_), y_min(y_min_), y_max(y_max_), z_min(z_min_), z_max(z_max_) {}

  __device__ __host__ inline BoundingBox_3D(
    volatile uint16_t& x_min_,
    volatile uint16_t& x_max_,
    volatile uint16_t& y_min_,
    volatile uint16_t& y_max_,
    volatile uint16_t& z_min_,
    volatile uint16_t& z_max_
) : x_min(x_min_), x_max(x_max_), y_min(y_min_), y_max(y_max_), z_min(z_min_), z_max(z_max_) {}

  __device__ __host__ inline BoundingBox_3D(const BoundingBox_3D& other) {
    x_min = other.x_min;
    x_max = other.x_max;
    y_min = other.y_min;
    y_max = other.y_max;
    z_min = other.z_min;
    z_max = other.z_max;
  };

  __device__ __host__ inline void operator=(const BoundingBox_3D& other) volatile {
    x_min = other.x_min;
    x_max = other.x_max;
    y_min = other.y_min;
    y_max = other.y_max;
    z_min = other.z_min;
    z_max = other.z_max;
  };

  __device__ __host__ inline void operator=(const BoundingBox_3D& other) {
    x_min = other.x_min;
    x_max = other.x_max;
    y_min = other.y_min;
    y_max = other.y_max;
    z_min = other.z_min;
    z_max = other.z_max;
  };
};

template<typename T>
class BoundingBoxOP_3D {
  public:
  static __device__ __host__ BoundingBox_3D apply(const BoundingBox_3D a,
                                                  const BoundingBox_3D b){
    return BoundingBox_3D(
      (uint16_t)min(a.x_min,b.x_min),
      (uint16_t)max(a.x_max,b.x_max),
      (uint16_t)min(a.y_min,b.y_min),
      (uint16_t)max(a.y_max,b.y_max),
      (uint16_t)min(a.z_min,b.z_min),
      (uint16_t)max(a.z_max,b.z_max)
    );
  };
  static __device__ __host__ bool equals(const BoundingBox_3D a,
                                        const BoundingBox_3D b){
    return a.x_min == b.x_min
        && a.x_max == b.x_max
        && a.y_min == b.y_min
        && a.y_min == b.y_min
        && a.z_min == b.z_max
        && a.z_max == b.z_max;
  };

  static __device__ __host__ BoundingBox_3D identity(){
    return BoundingBox_3D();
  };
  static __device__ __host__ BoundingBox_3D remove_volatile(volatile BoundingBox_3D& a){
    BoundingBox_3D temp = {
      a.x_min,
      a.x_max,
      a.y_min,
      a.y_max,
      a.z_min,
      a.z_max
    };

    return temp;
  };

  static __device__ __host__ BoundingBox_3D map_to(const T value,
                                                   const uint64_t flat_index,
                                                   const Space<3> space
                                                   ){

    if (value){
      Index<3> index(flat_index, space);
      return BoundingBox_3D(
        (uint16_t)index.x(),
        (uint16_t)index.x(),
        (uint16_t)index.y(),
        (uint16_t)index.y(),
        (uint16_t)index.z(),
        (uint16_t)index.z()
      );
    }

    return identity();
  };
};

template<typename T,  uint8_t CHUNK>
pybind11::list bounding_box(pybind11::array_t<T, ARRAY_FLAGS> arr){
  pybind11::buffer_info buffer = arr.request(false);
  if (buffer.ndim != 3){
    throw std::runtime_error("This function requires 3 dimensional input!");
  }

  const ssize_t items = buffer.size;
  if(items < 1){
    throw std::runtime_error("Invalid number of values in buffer");
  }

  const size_t buffer_size = items;
  BoundingBox_3D out;
  Space<3> space(
    buffer.shape[0], buffer.shape[1], buffer.shape[2]
  );

  cudaError_t error = reduce<1, BoundingBoxOP_3D<T>, T, BoundingBox_3D, Space<3>>(
    (T*)buffer.ptr,
    buffer_size,
    &out,
    {buffer.shape[0], buffer.shape[1], buffer.shape[2]}
  );

  if(error != cudaSuccess){
    std::cout << cudaGetErrorName(error) << "\n";
    std::cout << cudaGetErrorString(error) << "\n";
  }


  pybind11::list returnList(6);
  returnList[0] = out.x_min;
  returnList[1] = out.x_max;
  returnList[2] = out.y_min;
  returnList[3] = out.y_max;
  returnList[4] = out.z_min;
  returnList[5] = out.z_max;
  return returnList;
}

void apply_bounding_box_module(pybind11::module& m){
  m.def("bounding_box", &bounding_box<float, 1>);
}
