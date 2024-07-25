#pragma once

// This is strictly not needed
#include<functional>

template<typename... Ts>
void free_device_memory(Ts** && ... device_pointer);

class CudaRunner{
  public:
    CudaRunner(std::function<void(cudaError_t)> error_lambda);
    cudaError_t error() const;
    CudaRunner& operator|(std::function<cudaError_t()> func);
};
