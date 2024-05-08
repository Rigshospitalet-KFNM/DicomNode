/* This module setup work around CUDA different devices, allowing for better
fitting to the actual GPU device
*/
#ifndef DICOMNODE_CUDA_MANAGEMENT
#define DICOMNODE_CUDA_MANAGEMENT

#include<stdint.h>

#include<pybind11/pybind11.h>

namespace py = pybind11;

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
int32_t maximize_shared_memory(R (func)(Args...)){
  cudaDeviceProp dev_prop = get_current_device();
  if (dev_prop.major >= 7){
    cudaFuncSetAttribute(func, cudaFuncAttributeMaxDynamicSharedMemorySize, dev_prop.sharedMemPerBlockOptin);
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


#endif