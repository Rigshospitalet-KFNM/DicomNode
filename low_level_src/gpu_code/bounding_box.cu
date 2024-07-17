#ifndef DICOMNODE_BOUNDING_BOX
#define DICOMNODE_BOUNDING_BOX

// Standard library
#include<stdint.h>
#include<functional>
#include<iostream>

// Thrid party
#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>
#include<cuda_runtime.h>

// Dicomnode imports
#include"cuda_management.cu"
#include"indexing.cu"
#include"scan_reduce.cu"

constexpr uint8_t MAX_DIM = 8;


struct BoundingBox_3D {
  uint16_t x_min;
  uint16_t x_max;
  uint16_t y_min;
  uint16_t y_max;
  uint16_t z_min;
  uint16_t z_max;
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
  static __device__ __host__ BoundingBox_3D remove_volatile(volatile BoundingBox_3D a){
    return {
      .x_min = a.x_min,
      .x_max = a.x_max,
      .y_min = a.y_min,
      .y_max = a.y_max,
      .z_min = a.z_min,
      .z_max = a.z_max,
    };
  };
  
  static __device__ __host__ BoundingBox_3D map_to(T value, 
                                                   uint64_t index, 
                                                   uint16_t dim_x,
                                                   uint16_t dim_y,
                                                   uint16_t dim_z){
                                                    
    if (value){
      return {
        .x_min = (uint16_t)(index % dim_x),
        .x_max = (uint16_t)(index % dim_x),
        .y_min = (uint16_t)((index % ((uint32_t)dim_x * dim_y)) / dim_x),
        .y_max = (uint16_t)((index % ((uint32_t)dim_x * dim_y)) / dim_x),
        .z_min = (uint16_t)(index / ((uint32_t)dim_x * dim_y)),
        .z_max = (uint16_t)(index / ((uint32_t)dim_x * dim_y)),
      };
    }


    return *this::identity();
  }
};


template<typename T>
pybind11::list bounding_box(pybind11::array_t<T, pybind11::array::c_style 
                                               | pybind11::array::forcecast>){
  
}

void apply_bounding_box_module(pybind11::module& m){

}


#endif