// Pybind includes
#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>

#include"utilities.cuh"

#include"../gpu_code/dicom_node_gpu.cu"


pybind11::object cast_current_device(){
  return pybind11::cast(get_current_device());
}

void apply_cuda_management_module(pybind11::module& m){
  pybind11::class_<cudaDeviceProp>(m, "DeviceProperties")
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
    .def_readonly("directManagedMemAccessFromHost", &cudaDeviceProp::directManagedMemAccessFromHost)
    .def("__repr__",
    [](const cudaDeviceProp& prop){
      std::stringstream ss;
      ss << "-----Cuda Device Properties-----\n"
         << "Name: " << prop.name << "\n"
         << "Compute capability: " << prop.major << "." << prop.minor << "\n"
         << "Total memory: " << get_byte_string(prop.totalGlobalMem) << "\n"
         << "Shared memory: " << get_byte_string(prop.sharedMemPerBlock) << "\n"
         << "Shared memory (optin): " << get_byte_string(prop.sharedMemPerBlock) << "\n"
         << "Registers per block: " << get_byte_string(prop.regsPerBlock) << "\n"
         << "Registers per multiprocessor: " << get_byte_string(prop.regsPerMultiprocessor);
      return ss.str();
    });

  m.def("get_device_properties", &cast_current_device);
}
