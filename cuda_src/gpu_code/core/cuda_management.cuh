#pragma once

#include<concepts>
#include<functional>
#include<tuple>
#include<stack>
#include"error.cuh"

inline bool is_host_pointer(const cudaPointerAttributes& attr){
  return
       attr.type == cudaMemoryType::cudaMemoryTypeUnregistered
    || attr.type == cudaMemoryType::cudaMemoryTypeHost;
}

inline bool is_host_pointer(const void* ptr){
  cudaPointerAttributes attr{};
  cudaPointerGetAttributes(&attr, ptr); // Fucked if this fails
  return is_host_pointer(attr);
}

template<typename... Ts>
void free_device_memory(Ts** && ... device_pointer){
  ([&]{
    cudaError_t error = cudaFree(*device_pointer);
    if(error != cudaSuccess){
      const char* error_name = cudaGetErrorName(error);
      const char* error_desc = cudaGetErrorString(error);
      printf("Freeing %p failed with %s - %s\n", *device_pointer, error_name, error_desc);
    }
    *device_pointer = nullptr;
  }(), ...);
}

struct GPUAllocation {
  void* data = nullptr;
  u64 size = 0;

  GPUAllocation() = default;
  GPUAllocation(void* data_, const u64 size_ ) noexcept : data(data_), size(size_) {}
  GPUAllocation(const GPUAllocation&) = delete;
  GPUAllocation& operator=(const GPUAllocation&) = delete;
  GPUAllocation(GPUAllocation&& other) noexcept : data(other.data), size(other.size)  {
    other.data = nullptr;
    other.size = 0;
  }
  GPUAllocation& operator=(GPUAllocation&& other) noexcept {
    assert(data == nullptr);

    data = other.data;
    size = other.size;

    other.data = nullptr;

    return *this;
  }
};

template<typename T>
struct GPUSpan {
  T* data = nullptr;
  u64 size = 0;
};

class GPUAllocator {
  std::vector<GPUAllocation> allocations;

public:
  GPUAllocator() = default;
  GPUAllocator(const GPUAllocator&) = delete;
  GPUAllocator& operator=(const GPUAllocator&) = delete;
  GPUAllocator(GPUAllocator&& other) noexcept = default;
  GPUAllocator& operator=(GPUAllocator&& other) noexcept = default;

  ~GPUAllocator() noexcept {
    for (const auto& allocation : allocations) {
      cudaFree(allocation.data);
    }
  }

  cudaError_t allocate(GPUAllocation& requested_allocation) {
    for (auto& allocation : allocations) {
      if (allocation.size == requested_allocation.size) {
        requested_allocation = std::move(allocation);
        allocation = std::move(allocations.back());
        allocations.pop_back(); // It's ok to invalidate the iterator here
        return cudaSuccess;
      }
    }

    return cudaMalloc(&requested_allocation.data, requested_allocation.size);
  }

  void release(GPUAllocation& allocation) {
    allocations.emplace_back(std::move(allocation));
  }
};

class DicomNodeRunner{
  public:
    DicomNodeRunner() : m_error_function(
      []([[maybe_unused]] dicomNodeError_t _){}
    ){}

    DicomNodeRunner(std::function<void(dicomNodeError_t)> error_function)
    : m_error_function(std::move(error_function)) {}

  DicomNodeRunner& operator|(const std::function<cudaError_t()>& func){
    if(m_error == dicomNodeError_t::SUCCESS) [[likely]]{
      const cudaError_t ret = func();
      if(ret != cudaSuccess) [[unlikely]] {
        m_error = encode_cuda_error(ret);
        m_error_function(m_error);
      }
    }
    return *this;
  }

  DicomNodeRunner& operator|(const std::function<dicomNodeError_t()>& func){
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

  [[nodiscard]] constexpr dicomNodeError_t error() const noexcept {
    return m_error;
  }

  template<typename T>
  GPUSpan<T> allocate(u64 size) {
    GPUAllocation alloc;
    alloc.size = size;

    if (m_error == dicomNodeError_t::SUCCESS) {
      m_error = encode_cuda_error(allocator.allocate(alloc));
    }

    return GPUSpan<T>{
      .data = (T*)alloc.data,
      .size = size
    };
  }

  template<typename T>
  void release(GPUSpan<T> gpu_span) {
    GPUAllocation alloc{
      (void*)gpu_span.data,
      gpu_span.size
    };

    allocator.release(alloc);
  }

  private:
    std::function<void(dicomNodeError_t)> m_error_function;
    dicomNodeError_t m_error = dicomNodeError_t::SUCCESS;
    GPUAllocator allocator;
};

static std::tuple<dicomNodeError_t, cudaDeviceProp> get_current_device(){
  DicomNodeRunner runner;
  cudaDeviceProp prop = {};
  i32 current_device;
  runner
    | [&](){
      return cudaGetDevice(&current_device);
    } | [&](){
      return cudaGetDeviceProperties(&prop, current_device);
    };
  return {runner.error(), prop};
}

template<class R, class... Args>
u64 maximize_shared_memory(R (kernel)(Args...)){
  auto [error, dev_prop] = get_current_device();
  if (error == dicomNodeError_t::SUCCESS && dev_prop.major >= 7){
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dev_prop.sharedMemPerBlockOptin);
  }
  return dev_prop.sharedMemPerBlockOptin;
}

/** A monadic operation, that is different dependent on the
 *
 * @tparam HostFunction The function that should run if the pointer is a host pointer
 * @tparam DeviceFunction The function that should run if the pointer is a device pointer
 * @tparam PTR_T The Type of pointer
 * @tparam Args Arguments for the functions. Note that both functions should have the same arguments
 * @param pointer A nonnull host or device pointer
 * @param args Arguments for the
 * @return
 */
template<auto HostFunction, auto DeviceFunction, typename PTR_T, typename... Args>
  //requires std::invocable<decltype(HostFunction), PTR_T, Args...> &&
  //         std::invocable<decltype(DeviceFunction), PTR_T, Args...> &&
  //         std::same_as<std::invoke_result_t<decltype(HostFunction)>, dicomNodeError_t> &&
  //         std::same_as<std::invoke_result_t<decltype(DeviceFunction)>, dicomNodeError_t>
dicomNodeError_t pointer_location_dependent_code(PTR_T* pointer, Args... args) {
  DicomNodeRunner runner;
  cudaPointerAttributes attr{};

  runner | [&]() {
    return cudaPointerGetAttributes(&attr, pointer);
  } | [&](){
    dicomNodeError_t error;
    if (is_host_pointer(attr)) {
      error = HostFunction(pointer, args...);
    } else {
      error = DeviceFunction(pointer, args...);
    }
    return error;
  };

  return runner.error();
}

/** A monadic operation, that is different dependent on the
 *
 * @tparam HostFunction The function that should run if the pointer is a host pointer
 * @tparam DeviceFunction The function that should run if the pointer is a device pointer
 * @tparam PTR_T The Type of pointer
 * @tparam Args Arguments for the functions. Note that both functions should have the same arguments
 * @param pointer A nonnull host or device pointer
 * @param args Arguments for the
 * @return
 */
template<auto HostFunction, auto DeviceFunction, typename PTR_T, typename... Args>
  //requires std::invocable<decltype(HostFunction), PTR_T, Args...> &&
  //         std::invocable<decltype(DeviceFunction), PTR_T, Args...> &&
  //         std::same_as<std::invoke_result_t<decltype(HostFunction)>, dicomNodeError_t> &&
  //         std::same_as<std::invoke_result_t<decltype(DeviceFunction)>, dicomNodeError_t>
dicomNodeError_t pointer_location_dependent_code(DicomNodeRunner& runner, PTR_T* pointer, Args... args) {
  cudaPointerAttributes attr;

  runner | [&]() {
    return cudaPointerGetAttributes(&attr, pointer);
  } | [&](){
    dicomNodeError_t error;
    if (is_host_pointer(attr)) {
      error = HostFunction(pointer, args...);
    } else {
      error = DeviceFunction(pointer, args...);
    }
    return error;
  };

  return runner.error();
}


static __device__ u64 get_gid(){
  return blockDim.x * blockIdx.x + threadIdx.x;
}
