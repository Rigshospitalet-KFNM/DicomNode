#ifndef DICOMNODE_CUDA_MANAGEMENT_H
#define DICOMNODE_CUDA_MANAGEMENT_H

#include<stdint.h>
#include<functional>
#include<pybind11/pybind11.h>

namespace py = pybind11;

/**
 * @brief Get the current device object
 *
 * @return cudaDeviceProp of the current device
 */
cudaDeviceProp get_current_device();

template<class R, class... Args>
int32_t maximize_shared_memory(R (kernel)(Args...));

/**
 * @brief
 *
 * @tparam Ts 
 * @param device_pointer 
 */
template<typename... Ts>
void free_device_memory(Ts** && ... device_pointer);



void run_cuda(std::function<cudaError_t()> action_function,
              std::function<void(cudaError_t)> error_function);

void apply_cuda_management_module(py::module& m);

#endif