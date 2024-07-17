#pragma once

// Thrid party
#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>

// Dicomnode imports
#include"core/core.cu"
#include"scan_reduce.cu"

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
  static __device__ __host__ BoundingBox_3D apply(BoundingBox_3D a,
                                                  BoundingBox_3D b){
    return {
      .x_min = (uint16_t)min(a.x_min,b.x_min),
      .x_max = (uint16_t)max(a.x_max,b.x_max),
      .y_min = (uint16_t)min(a.y_min,b.y_min),
      .y_max = (uint16_t)max(a.y_max,b.y_max),
      .z_min = (uint16_t)min(a.z_min,b.z_min),
      .z_max = (uint16_t)max(a.z_max,b.z_max),
    };
  };
  static __device__ __host__ bool equal(BoundingBox_3D a,
                                        BoundingBox_3D b){
    return a.x_min == b.x_min
        && a.x_max == b.x_max
        && a.y_min == b.y_min
        && a.y_min == b.y_min
        && a.z_min == b.z_max
        && a.z_max == b.z_max;
  };

  static __device__ __host__ BoundingBox_3D identity(){
    return {
      .x_min = UINT16_MAX,
      .x_max = 0,
      .y_min = UINT16_MAX,
      .y_max = 0,
      .z_min = UINT16_MAX,
      .z_max = 0
    };
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

  static __device__ __host__ BoundingBox_3D map_to(T value,
                                                   uint64_t index,
                                                   Space<3> space
                                                   ){

    if (value){
      return space.from_flat_index(index);
    }

    return identity();
  }
};

template<typename T>
pybind11::list bounding_box(pybind11::array_t<T, array_flags>){

}

void apply_bounding_box_module(pybind11::module& m){

}
