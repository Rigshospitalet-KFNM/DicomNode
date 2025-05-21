#pragma once

// This is strictly not needed
#include<functional>
#include<iostream>
#include<tuple>
#include"error.cuh"

static bool is_host_pointer(const cudaPointerAttributes& attr){
  return
       attr.type == cudaMemoryType::cudaMemoryTypeUnregistered
    || attr.type == cudaMemoryType::cudaMemoryTypeHost;
}

static bool is_host_pointer(const void* ptr){
  cudaPointerAttributes attr;
  cudaPointerGetAttributes(&attr, ptr); // Fucked if this fails
  return is_host_pointer(attr);
}

template<typename... Ts>
void free_device_memory(Ts** && ... device_pointer){
  ([&]{
    if(!(*device_pointer)){
      return;
    }

    if(!is_host_pointer(device_pointer)){
      cudaError_t error = cudaFree(*device_pointer);
      if(error != cudaSuccess){
        const char* error_name = cudaGetErrorName(error);
        const char* error_desc = cudaGetErrorString(error);
        printf("Freeing %p failed with %s - %s\n", *device_pointer, error_name, error_desc);
      }
      *device_pointer = nullptr;
    }
  }(), ...);
}

class CudaRunner{
  public:
    CudaRunner(std::function<void(cudaError_t)> error_function)
      : m_error_function(error_function){}

    CudaRunner()
      : m_error_function([](cudaError_t error){}){}

    cudaError_t error() const {
      return m_error;
    }

    CudaRunner& operator|(std::function<cudaError_t()> func) {
      if(m_error == cudaSuccess) [[likely]] {
        m_error = func();
        if (m_error != cudaSuccess) [[unlikely]] {
          m_error_function(m_error);
        }
      }
      return *this;
    };

    template<typename F>
      requires std::invocable<F> &&
               std::same_as<std::invoke_result_t<F>, cudaError_t>
    CudaRunner& operator|(F&& func) {
      if(m_error == cudaSuccess) [[likely]]{
        m_error = func();
        if (m_error != cudaSuccess)[[unlikely]]{
          m_error_function(m_error);
        }
      }
      return *this;
    };

  private:
    std::function<void(cudaError_t)> m_error_function;
    cudaError_t m_error = cudaSuccess;
};

class DicomNodeRunner{
  public:
    DicomNodeRunner() : m_error_function([](dicomNodeError_t _){}){}

    DicomNodeRunner(std::function<void(dicomNodeError_t)> error_funciton)
    : m_error_function(error_funciton) {}

  DicomNodeRunner& operator|(std::function<cudaError_t()> func){
    if(m_error == dicomNodeError_t::SUCCESS) [[likely]]{
      cudaError_t ret = func();
      if(ret != cudaSuccess) [[unlikely]] {
        m_error = encode_cuda_error(ret);
        m_error_function(m_error);
      }
    }
    return *this;
  }

  DicomNodeRunner& operator|(std::function<dicomNodeError_t()> func){
    if(m_error == dicomNodeError_t::SUCCESS) [[likely]]{
      m_error = func();
      if(m_error != dicomNodeError_t::SUCCESS) [[unlikely]]{
        m_error_function(m_error);
      }
    }
    return *this;
  }

  template<typename F>
    requires std::invocable<F> &&
             std::same_as<std::invoke_result_t<F>, dicomNodeError_t>
  DicomNodeRunner& operator|(F&& func){
    if(m_error == dicomNodeError_t::SUCCESS) [[likely]]{
      m_error = func();
      if(m_error != dicomNodeError_t::SUCCESS)[[unlikely]]{
        m_error_function(m_error);
      }
    }
    return *this;
  }

  dicomNodeError_t error() const {
    return m_error;
  }

  private:
    std::function<void(dicomNodeError_t)> m_error_function;
    dicomNodeError_t m_error = dicomNodeError_t::SUCCESS;
};

static std::tuple<cudaError_t, cudaDeviceProp> get_current_device(){
  CudaRunner runner;
  cudaDeviceProp prop;
  int current_device;
  runner
    | [&](){
      return cudaGetDevice(&current_device);
    } | [&](){
      return cudaGetDeviceProperties(&prop, current_device);
    };
  return {runner.error(), prop};
}

template<class R, class... Args>
int32_t maximize_shared_memory(R (kernel)(Args...)){
  auto [error, dev_prop] = get_current_device();
  if (error == cudaSuccess && dev_prop.major >= 7){
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dev_prop.sharedMemPerBlockOptin);
  }
  return dev_prop.sharedMemPerBlockOptin;
}
