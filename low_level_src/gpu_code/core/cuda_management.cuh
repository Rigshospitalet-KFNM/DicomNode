#pragma once

// This is strictly not needed
#include<functional>
#include<iostream>

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

cudaDeviceProp get_current_device();

class CudaRunner{
  public:
    CudaRunner(std::function<void(cudaError_t)> error_lambda)
      : m_error_function(error_lambda){}

    cudaError_t error() const {
      return m_error;
    }

    CudaRunner& operator|(std::function<cudaError_t()> func) {
      if(m_error == cudaSuccess){
        m_error = func();
        if (m_error != cudaSuccess){
          m_error_function(m_error);
        }
      }
      return *this;
    };

  private:
    std::function<void(cudaError_t)> m_error_function;
    cudaError_t m_error = cudaSuccess;
};

cudaDeviceProp get_current_device(){
  cudaDeviceProp prop;
  int current_device;
  cudaGetDevice(&current_device);
  cudaGetDeviceProperties(&prop, current_device);
  return prop;
}

template<class R, class... Args>
int32_t maximize_shared_memory(R (kernel)(Args...)){
  cudaDeviceProp dev_prop = get_current_device();
  if (dev_prop.major >= 7){
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dev_prop.sharedMemPerBlockOptin);
  }
  return dev_prop.sharedMemPerBlockOptin;
}
