#ifndef DICOMNODE_CUDA_MIRROR
#define DICOMNODE_CUDA_MIRROR

// Standard library
#include<stdint.h>
#include<functional>

// Thrid party
#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>

// Dicomnode imports
#include"cuda_management.cu"

template<typename T>
__device__ T index3D(T x, T y, T z, T x_dim, T y_max, T z_dim){
  return z * x_dim * y_max
       + y * x_dim
       + x;
}


template<typename T>
__device__ T invert_index(T index, T size){
  return size - index - 1;
}

template<typename T>
__global__ void kernel_3D_mirror_X(T* data_in,
                                   T* data_out,
                                   uint32_t x_dim,
                                   uint32_t y_dim,
                                   uint32_t z_dim){
  uint32_t global_index = blockDim.x * blockIdx.x + threadIdx.x;
  uint32_t x_in = global_index % x_dim;
  uint32_t y_in = global_index / x_dim % y_dim;
  uint32_t z_in = global_index / (x_dim * y_dim);

  uint32_t x_out = invert_index(x_in, x_dim);
  uint32_t y_out = y_in;
  uint32_t z_out = z_in;


  if (global_index < x_dim * y_dim * z_dim){
    data_out[index3D(x_out, y_out, z_out, x_dim, y_dim, z_dim)] =
      data_in[index3D(x_in, y_in, z_in, x_dim, y_dim, z_dim)];
  }
}

template<typename T>
__global__ void kernel_3D_mirror_Y(T* data_in,
                                   T* data_out,
                                   uint32_t x_dim,
                                   uint32_t y_dim,
                                   uint32_t z_dim){
  uint32_t global_index = blockDim.x * blockIdx.x + threadIdx.x;
  uint32_t x_in = global_index % x_dim;
  uint32_t y_in = global_index / x_dim % y_dim;
  uint32_t z_in = global_index / (x_dim * y_dim);

  uint32_t x_out = x_in;
  uint32_t y_out = invert_index(y_in, y_dim);
  uint32_t z_out = z_in;

  if (global_index < x_dim * y_dim * z_dim){
    data_out[index3D(x_out, y_out, z_out, x_dim, y_dim, z_dim)] =
      data_in[index3D(x_in, y_in, z_in, x_dim, y_dim, z_dim)];
  }
}

template<typename T>
__global__ void kernel_3D_mirror_Z(T* data_in,
                                   T* data_out,
                                   uint32_t x_dim,
                                   uint32_t y_dim,
                                   uint32_t z_dim){
  uint32_t global_index = blockDim.x * blockIdx.x + threadIdx.x;
  uint32_t x_in = global_index % x_dim;
  uint32_t y_in = global_index / x_dim % y_dim;
  uint32_t z_in = global_index / (x_dim * y_dim);

  uint32_t x_out = x_in;
  uint32_t y_out = y_in;
  uint32_t z_out = invert_index(z_in, z_dim);

  if (global_index < x_dim * y_dim * z_dim){
    data_out[index3D(x_out, y_out, z_out, x_dim, y_dim, z_dim)] =
      data_in[index3D(x_in, y_in, z_in, x_dim, y_dim, z_dim)];
  }
}

template<typename T>
__global__ void kernel_3D_mirror_XY(T* data_in,
                                    T* data_out,
                                    uint32_t x_dim,
                                    uint32_t y_dim,
                                    uint32_t z_dim){
  uint32_t global_index = blockDim.x * blockIdx.x + threadIdx.x;
  uint32_t x_in = global_index % x_dim;
  uint32_t y_in = global_index / x_dim % y_dim;
  uint32_t z_in = global_index / (x_dim * y_dim);

  uint32_t x_out = invert_index(x_in, x_dim);
  uint32_t y_out = invert_index(y_in, y_dim);
  uint32_t z_out = z_in;


  if (global_index < x_dim * y_dim * z_dim){
    data_out[index3D(x_out, y_out, z_out, x_dim, y_dim, z_dim)] =
      data_in[index3D(x_in, y_in, z_in, x_dim, y_dim, z_dim)];
  }
}

template<typename T>
__global__ void kernel_3D_mirror_XZ(T* data_in,
                                    T* data_out,
                                    uint32_t x_dim,
                                    uint32_t y_dim,
                                    uint32_t z_dim){
  uint32_t global_index = blockDim.x * blockIdx.x + threadIdx.x;
  uint32_t x_in = global_index % x_dim;
  uint32_t y_in = global_index / x_dim % y_dim;
  uint32_t z_in = global_index / (x_dim * y_dim);

  uint32_t x_out = invert_index(x_in, x_dim);
  uint32_t y_out = y_in;
  uint32_t z_out = invert_index(z_in, z_dim);


  if (global_index < x_dim * y_dim * z_dim){
    data_out[index3D(x_out, y_out, z_out, x_dim, y_dim, z_dim)] =
      data_in[index3D(x_in, y_in, z_in, x_dim, y_dim, z_dim)];
  }
}

template<typename T>
__global__ void kernel_3D_mirror_YZ(T* data_in,
                                    T* data_out,
                                    uint32_t x_dim,
                                    uint32_t y_dim,
                                    uint32_t z_dim){
  uint32_t global_index = blockDim.x * blockIdx.x + threadIdx.x;
  uint32_t x_in = global_index % x_dim;
  uint32_t y_in = global_index / x_dim % y_dim;
  uint32_t z_in = global_index / (x_dim * y_dim);

  uint32_t x_out = x_in;
  uint32_t y_out = invert_index(y_in, y_dim);
  uint32_t z_out = invert_index(z_in, z_dim);


  if (global_index < x_dim * y_dim * z_dim){
    data_out[index3D(x_out, y_out, z_out, x_dim, y_dim, z_dim)] =
      data_in[index3D(x_in, y_in, z_in, x_dim, y_dim, z_dim)];
  }
}

template<typename T>
__global__ void kernel_3D_mirror_XYZ(T* data_in,
                                     T* data_out,
                                     uint32_t x_dim,
                                     uint32_t y_dim,
                                     uint32_t z_dim){
  uint32_t global_index = blockDim.x * blockIdx.x + threadIdx.x;
  uint32_t x_in = global_index % x_dim;
  uint32_t y_in = global_index / x_dim % y_dim;
  uint32_t z_in = global_index / (x_dim * y_dim);

  uint32_t x_out = invert_index(x_in, x_dim);
  uint32_t y_out = invert_index(y_in, y_dim);
  uint32_t z_out = invert_index(z_in, z_dim);


  if (global_index < x_dim * y_dim * z_dim){
    data_out[index3D(x_out, y_out, z_out, x_dim, y_dim, z_dim)] =
      data_in[index3D(x_in, y_in, z_in, x_dim, y_dim, z_dim)];
  }
}

// py::array::c_style | py::array::forcecast ensures that the array is a dense
// C style array, I should consider overloading to F style as it's just indexing
// And the depth is O(1)
template<typename T, void (*kernel)(T*, T*, uint32_t, uint32_t, uint32_t)>
int map3D(pybind11::array_t<T, pybind11::array::c_style
                             | pybind11::array::forcecast> arr){
  pybind11::buffer_info arr_buffer = arr.request(true);
  uint64_t buffer_size = arr_buffer.size * sizeof(T);

  if(arr_buffer.ndim != 3){
    throw std::runtime_error("Input shape must be 3");
  }

  uint32_t threads = arr_buffer.shape[0]
                   * arr_buffer.shape[1]
                   * arr_buffer.shape[2];

  uint32_t threads_per_block = 1024;
  uint32_t blocks = threads % threads_per_block == 0
                ? threads / threads_per_block
                : threads / threads_per_block + 1;

  // Initial success value is unused, however I dislike uninitailized variables
  // Device points
  T* dev_in = nullptr;
  T* dev_out = nullptr;
  // We allocate once, and index into dev_in to place dev_in
  auto error_function = [&](cudaError_t error){
    free_device_memory(&dev_in);
  };
  CudaRunner runner{error_function};
  runner | [&](){return cudaMalloc(&dev_in, 2 * buffer_size);};
  if(runner.error != cudaSuccess){
    return (int)runner.error;
  }
  dev_out = dev_in + arr_buffer.size;
  runner
    | [&](){return cudaMemcpy(dev_in, arr_buffer.ptr, buffer_size, cudaMemcpyHostToDevice);}
    | [&](){
      kernel<<<blocks,threads_per_block>>>(dev_in,
                                       dev_out,
                                       arr_buffer.shape[2],
                                       arr_buffer.shape[1],
                                       arr_buffer.shape[0]);
      return cudaGetLastError();}
    | [&](){
      return cudaMemcpy(arr_buffer.ptr, dev_out, buffer_size,
                       cudaMemcpyDeviceToHost);
    };
  // I assume you can always free, this might be wrong
  cudaFree(dev_in);

  return (int)runner.error;
}

void apply_mirror_module(pybind11::module& m){
  const char* mirror_x_name = "mirror_x";
  const char* mirror_y_name = "mirror_y";
  const char* mirror_z_name = "mirror_z";
  const char* mirror_xy_name = "mirror_xy";
  const char* mirror_xz_name = "mirror_xz";
  const char* mirror_yz_name = "mirror_yz";
  const char* mirror_xyz_name = "mirror_xyz";

  const char* mirror_x_doc = "Mirror as 3D volume along the X axis";
  const char* mirror_y_doc = "Mirror as 3D volume along the Y axis";
  const char* mirror_z_doc = "Mirror as 3D volume along the Z axis";
  const char* mirror_xy_doc = "Mirror as 3D volume along the X axis and the Y axis";
  const char* mirror_xz_doc = "Mirror as 3D volume along the X axis and the Z axis";
  const char* mirror_yz_doc = "Mirror as 3D volume along the Y axis and the Z axis";
  const char* mirror_xyz_doc = "Mirror as 3D volume along the X,Y,Z axis";

  m.def(mirror_x_name, &map3D<double  , kernel_3D_mirror_X>, mirror_x_doc);
  m.def(mirror_x_name, &map3D<float   , kernel_3D_mirror_X>, mirror_x_doc);
  m.def(mirror_x_name, &map3D<int8_t  , kernel_3D_mirror_X>, mirror_x_doc);
  m.def(mirror_x_name, &map3D<int16_t , kernel_3D_mirror_X>, mirror_x_doc);
  m.def(mirror_x_name, &map3D<int32_t , kernel_3D_mirror_X>, mirror_x_doc);
  m.def(mirror_x_name, &map3D<int64_t , kernel_3D_mirror_X>, mirror_x_doc);
  m.def(mirror_x_name, &map3D<uint8_t , kernel_3D_mirror_X>, mirror_x_doc);
  m.def(mirror_x_name, &map3D<uint16_t, kernel_3D_mirror_X>, mirror_x_doc);
  m.def(mirror_x_name, &map3D<uint32_t, kernel_3D_mirror_X>, mirror_x_doc);
  m.def(mirror_x_name, &map3D<uint64_t, kernel_3D_mirror_X>, mirror_x_doc);

  m.def(mirror_y_name, &map3D<double  , kernel_3D_mirror_Y>, mirror_y_doc);
  m.def(mirror_y_name, &map3D<float   , kernel_3D_mirror_Y>, mirror_y_doc);
  m.def(mirror_y_name, &map3D<int8_t  , kernel_3D_mirror_Y>, mirror_y_doc);
  m.def(mirror_y_name, &map3D<int16_t , kernel_3D_mirror_Y>, mirror_y_doc);
  m.def(mirror_y_name, &map3D<int32_t , kernel_3D_mirror_Y>, mirror_y_doc);
  m.def(mirror_y_name, &map3D<int64_t , kernel_3D_mirror_Y>, mirror_y_doc);
  m.def(mirror_y_name, &map3D<uint8_t , kernel_3D_mirror_Y>, mirror_y_doc);
  m.def(mirror_y_name, &map3D<uint16_t, kernel_3D_mirror_Y>, mirror_y_doc);
  m.def(mirror_y_name, &map3D<uint32_t, kernel_3D_mirror_Y>, mirror_y_doc);
  m.def(mirror_y_name, &map3D<uint64_t, kernel_3D_mirror_Y>, mirror_y_doc);

  m.def(mirror_z_name, &map3D<double  , kernel_3D_mirror_Z>, mirror_z_doc);
  m.def(mirror_z_name, &map3D<float   , kernel_3D_mirror_Z>, mirror_z_doc);
  m.def(mirror_z_name, &map3D<int8_t  , kernel_3D_mirror_Z>, mirror_z_doc);
  m.def(mirror_z_name, &map3D<int16_t , kernel_3D_mirror_Z>, mirror_z_doc);
  m.def(mirror_z_name, &map3D<int32_t , kernel_3D_mirror_Z>, mirror_z_doc);
  m.def(mirror_z_name, &map3D<int64_t , kernel_3D_mirror_Z>, mirror_z_doc);
  m.def(mirror_z_name, &map3D<uint8_t , kernel_3D_mirror_Z>, mirror_z_doc);
  m.def(mirror_z_name, &map3D<uint16_t, kernel_3D_mirror_Z>, mirror_z_doc);
  m.def(mirror_z_name, &map3D<uint32_t, kernel_3D_mirror_Z>, mirror_z_doc);
  m.def(mirror_z_name, &map3D<uint64_t, kernel_3D_mirror_Z>, mirror_z_doc);

  m.def(mirror_xy_name, &map3D<double  , kernel_3D_mirror_XY>, mirror_xy_doc);
  m.def(mirror_xy_name, &map3D<float   , kernel_3D_mirror_XY>, mirror_xy_doc);
  m.def(mirror_xy_name, &map3D<int8_t  , kernel_3D_mirror_XY>, mirror_xy_doc);
  m.def(mirror_xy_name, &map3D<int16_t , kernel_3D_mirror_XY>, mirror_xy_doc);
  m.def(mirror_xy_name, &map3D<int32_t , kernel_3D_mirror_XY>, mirror_xy_doc);
  m.def(mirror_xy_name, &map3D<int64_t , kernel_3D_mirror_XY>, mirror_xy_doc);
  m.def(mirror_xy_name, &map3D<uint8_t , kernel_3D_mirror_XY>, mirror_xy_doc);
  m.def(mirror_xy_name, &map3D<uint16_t, kernel_3D_mirror_XY>, mirror_xy_doc);
  m.def(mirror_xy_name, &map3D<uint32_t, kernel_3D_mirror_XY>, mirror_xy_doc);
  m.def(mirror_xy_name, &map3D<uint64_t, kernel_3D_mirror_XY>, mirror_xy_doc);

  m.def(mirror_xz_name, &map3D<double  , kernel_3D_mirror_XZ>, mirror_xz_doc);
  m.def(mirror_xz_name, &map3D<float   , kernel_3D_mirror_XZ>, mirror_xz_doc);
  m.def(mirror_xz_name, &map3D<int8_t  , kernel_3D_mirror_XZ>, mirror_xz_doc);
  m.def(mirror_xz_name, &map3D<int16_t , kernel_3D_mirror_XZ>, mirror_xz_doc);
  m.def(mirror_xz_name, &map3D<int32_t , kernel_3D_mirror_XZ>, mirror_xz_doc);
  m.def(mirror_xz_name, &map3D<int64_t , kernel_3D_mirror_XZ>, mirror_xz_doc);
  m.def(mirror_xz_name, &map3D<uint8_t , kernel_3D_mirror_XZ>, mirror_xz_doc);
  m.def(mirror_xz_name, &map3D<uint16_t, kernel_3D_mirror_XZ>, mirror_xz_doc);
  m.def(mirror_xz_name, &map3D<uint32_t, kernel_3D_mirror_XZ>, mirror_xz_doc);
  m.def(mirror_xz_name, &map3D<uint64_t, kernel_3D_mirror_XZ>, mirror_xz_doc);

  m.def(mirror_yz_name, &map3D<double  , kernel_3D_mirror_YZ>, mirror_yz_doc);
  m.def(mirror_yz_name, &map3D<float   , kernel_3D_mirror_YZ>, mirror_yz_doc);
  m.def(mirror_yz_name, &map3D<int8_t  , kernel_3D_mirror_YZ>, mirror_yz_doc);
  m.def(mirror_yz_name, &map3D<int16_t , kernel_3D_mirror_YZ>, mirror_yz_doc);
  m.def(mirror_yz_name, &map3D<int32_t , kernel_3D_mirror_YZ>, mirror_yz_doc);
  m.def(mirror_yz_name, &map3D<int64_t , kernel_3D_mirror_YZ>, mirror_yz_doc);
  m.def(mirror_yz_name, &map3D<uint8_t , kernel_3D_mirror_YZ>, mirror_yz_doc);
  m.def(mirror_yz_name, &map3D<uint16_t, kernel_3D_mirror_YZ>, mirror_yz_doc);
  m.def(mirror_yz_name, &map3D<uint32_t, kernel_3D_mirror_YZ>, mirror_yz_doc);
  m.def(mirror_yz_name, &map3D<uint64_t, kernel_3D_mirror_YZ>, mirror_yz_doc);

  m.def(mirror_xyz_name, &map3D<double  , kernel_3D_mirror_XYZ>, mirror_xyz_doc);
  m.def(mirror_xyz_name, &map3D<float   , kernel_3D_mirror_XYZ>, mirror_xyz_doc);
  m.def(mirror_xyz_name, &map3D<int8_t  , kernel_3D_mirror_XYZ>, mirror_xyz_doc);
  m.def(mirror_xyz_name, &map3D<int16_t , kernel_3D_mirror_XYZ>, mirror_xyz_doc);
  m.def(mirror_xyz_name, &map3D<int32_t , kernel_3D_mirror_XYZ>, mirror_xyz_doc);
  m.def(mirror_xyz_name, &map3D<int64_t , kernel_3D_mirror_XYZ>, mirror_xyz_doc);
  m.def(mirror_xyz_name, &map3D<uint8_t , kernel_3D_mirror_XYZ>, mirror_xyz_doc);
  m.def(mirror_xyz_name, &map3D<uint16_t, kernel_3D_mirror_XYZ>, mirror_xyz_doc);
  m.def(mirror_xyz_name, &map3D<uint32_t, kernel_3D_mirror_XYZ>, mirror_xyz_doc);
  m.def(mirror_xyz_name, &map3D<uint64_t, kernel_3D_mirror_XYZ>, mirror_xyz_doc);
}

#endif