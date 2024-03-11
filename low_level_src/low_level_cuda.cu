#ifndef LOW_LEVEL_CUDA_DICOMNODE_H
#define LOW_LEVEL_CUDA_DICOMNODE_H

#include<iostream>
#include<stdint.h>
#include<float.h>
#include<vector>
#include<string.h>
#include<typeinfo>

#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>


template<typename T>
__device__ T index3D(T x, T y, T z, T x_dim, T y_max ,T z_dim){
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
__global__ void kernel_mirror_Y(){}
__global__ void kernel_mirror_Z(){}
__global__ void kernel_mirror_XY(){}
__global__ void kernel_mirror_XZ(){}
__global__ void kernel_mirror_YZ(){}
__global__ void kernel_mirror_XYZ(){}


// py::array::c_style | py::array::forcecast ensures that the array is a dense
// C style array, I should consider overloading to F style as it's just indexing
// And the depth is O(1)

template<typename T>
pybind11::array_t<T> mirror_x(pybind11::array_t<T, pybind11::array::c_style
                                | pybind11::array::forcecast> arr){
  auto arr_buffer = arr.request();
  uint64_t buffer_size = arr_buffer.size * sizeof(T);

  T a = 0;
  auto t = arr.dtype();
  t.attr("name")

  std::cout << "Type: " << typeid(decltype(a)).name() << "\n";

  if(arr_buffer.ndim != 3){
    throw std::runtime_error("Input shape must be 3");
  }

  // Initial success value is unused, however I dislike uninitailized variables
  cudaError error = cudaSuccess;

  auto out = pybind11::array_t<T>(arr_buffer.shape);
  auto out_buffer = out.request();


  // Device points
  T* dev_in;
  T* dev_out;
  // Macro this away
  error = cudaMalloc(&dev_in, buffer_size);

  error = cudaMemcpy(dev_in, arr_buffer.ptr, buffer_size,
                     cudaMemcpyHostToDevice);

  error = cudaMalloc(&dev_out, buffer_size);

  uint32_t threads = arr_buffer.shape[0]
                   * arr_buffer.shape[1]
                   * arr_buffer.shape[2];

  uint32_t threads_per_block = 1024;
  uint32_t blocks = threads % threads_per_block == 0
                ? threads / threads_per_block
                : threads / threads_per_block + 1;

  kernel_3D_mirror_X<<<blocks,threads_per_block>>>(dev_in,
                                                   dev_out,
                                                   arr_buffer.shape[2],
                                                   arr_buffer.shape[1],
                                                   arr_buffer.shape[0]);

  error = cudaGetLastError();
  std::cout << "Error:" << (int)error << "\n";
  std::cout << "Size:" << (int)arr_buffer.size << "\n";

  error = cudaMemcpy(out_buffer.ptr, dev_out, buffer_size,
                     cudaMemcpyDeviceToHost);

  // I assume you can always free, this might be wrong
  cudaFree(dev_in);
  cudaFree(dev_out);

  return out;
}


int cuda_add(int i, int j){
  return i + j;
}

PYBIND11_MODULE(_cuda, m){
  m.doc() = "pybind11 example plugin";
  m.attr("__name__") = "dicomnode._cuda";

  const char* mirror_x_name = "mirror_x";
  const char* mirror_x_doc = "Mirror as 3D volume along the X axis";

  m.def(mirror_x_name, &mirror_x<int32_t>,    mirror_x_doc);
  m.def(mirror_x_name, &mirror_x<float>,  mirror_x_doc);
  m.def(mirror_x_name, &mirror_x<double>, mirror_x_doc);
  m.def("add", &cuda_add, "A function that adds two numbers");
}

#endif