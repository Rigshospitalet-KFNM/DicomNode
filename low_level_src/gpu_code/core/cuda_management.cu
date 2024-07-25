#pragma once

/* This module setup work around CUDA different devices, allowing for better
fitting to the actual GPU device
*/
// C includes
#include<stdint.h>
// STL includes
#include<iostream>
#include<string>
#include<functional>
// Pybind includes
#include<pybind11/pybind11.h>
namespace py = pybind11;

#if defined(__CUDACC__) // NVCC
   #define ALIGN(n) __align__(n)
#elif defined(__GNUC__) // GCC
  #define ALIGN(n) __attribute__((aligned(n)))
#elif defined(_MSC_VER) // MSVC
  #define ALIGN(n) __declspec(align(n))
#else
  #error "Please provide a definition for ALIGN macro for your host compiler!"
#endif

template<typename... Ts>
void free_device_memory(Ts** && ... device_pointer){
  ([&]{
    cudaPointerAttributes attr;
    cudaError_t error = cudaPointerGetAttributes(&attr, *device_pointer);
    if(error != cudaSuccess){
      std::cout << "something went wrong!\n";
      return;
    }
    if(attr.type == cudaMemoryType::cudaMemoryTypeDevice || attr.type == cudaMemoryType::cudaMemoryTypeManaged){
      error = cudaFree(*device_pointer);
      if(error != cudaSuccess){
        std::cout << "freeing failed!";
      }
      *device_pointer = nullptr;
    }
  }(), ...);
}

class CudaRunner {
  std::function<void(cudaError_t)> error_function;
  cudaError_t m_error = cudaSuccess;
  public:
    cudaError_t error() const {
      return m_error;
    }
    CudaRunner(std::function<void(cudaError_t)> error_lambda) : error_function(error_lambda){}
    CudaRunner& operator|(std::function<cudaError_t()> func){
       if(m_error == cudaSuccess){
        m_error = func();
        if (m_error != cudaSuccess){
          error_function(m_error);
        }
      }
      return *this;
    };
};

void run_cuda(std::function<cudaError_t()> action_function,
              std::function<void(cudaError_t)> error_function){
    cudaError error = action_function();
    if(error != cudaSuccess){
        error_function(error);
    }
}


cudaDeviceProp get_current_device(){
  cudaDeviceProp prop;
  int current_device;
  cudaGetDevice(&current_device);
  cudaGetDeviceProperties(&prop, current_device);
  return prop;
}

py::object cast_current_device(){
  return py::cast(get_current_device());
}

template<class R, class... Args>
int32_t maximize_shared_memory(R (kernel)(Args...)){
  cudaDeviceProp dev_prop = get_current_device();
  if (dev_prop.major >= 7){
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dev_prop.sharedMemPerBlockOptin);
  }
  return dev_prop.sharedMemPerBlockOptin;
}

void apply_cuda_management_module(py::module& m){
  py::class_<cudaDeviceProp>(m, "DeviceProperties")
    .def_readonly("major", &cudaDeviceProp::major)
    .def_readonly("minor", &cudaDeviceProp::minor)
    .def_readonly("totalGlobalMem", &cudaDeviceProp::totalGlobalMem)
    .def_readonly("totalConstMem", &cudaDeviceProp::totalConstMem)
    .def_readonly("name", &cudaDeviceProp::name)
    .def_readonly("mangedMemory", &cudaDeviceProp::managedMemory)
    .def_readonly("sharedMemPerBlock", &cudaDeviceProp::sharedMemPerBlock)
    .def_readonly("sharedMemPerBlockOptin", &cudaDeviceProp::sharedMemPerBlockOptin)
    .def_readonly("sharedMemPerMultiprocessor", &cudaDeviceProp::sharedMemPerMultiprocessor)
    .def_readonly("unifiedAddressing", &cudaDeviceProp::unifiedAddressing)
    .def_readonly("unifiedFunctionPointers", &cudaDeviceProp::unifiedFunctionPointers)
    .def_readonly("concurrentKernels", &cudaDeviceProp::concurrentKernels)
    .def_readonly("concurrentManagedAccess", &cudaDeviceProp::concurrentManagedAccess)
    .def_readonly("directManagedMemAccessFromHost", &cudaDeviceProp::directManagedMemAccessFromHost);

  m.def("get_device_properties", &cast_current_device);
}
