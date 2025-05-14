// Pybind includes
#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>

#include<tuple>

#include"utilities.cuh"

#include"../gpu_code/dicom_node_gpu.cuh"


pybind11::object cast_current_device(){
  return pybind11::cast(get_current_device());
}

void print_image(const pybind11::object& python_image){
  Image<3, float> host_image;
  dicomNodeError_t error = load_image(&host_image, python_image);

  if(error){
    std::cout << "Encoutered dicomnode Error:" << error << "\n";
  } else {
    std::cout << "Starting Point: (" << host_image.space.starting_point[0] << ", "
                                     << host_image.space.starting_point[1] << ", "
                                     << host_image.space.starting_point[2] << ")\n";
    std::cout << "Extent: (" << host_image.space.extent[0] << ", "
                             << host_image.space.extent[1] << ", "
                             << host_image.space.extent[2] << ")\n";
  }

  free_image(&host_image);
}
void apply_cuda_management_module(pybind11::module& m){
  pybind11::class_<cudaDeviceProp>(m, "DeviceProperties")
    .def_readonly("major", &cudaDeviceProp::major)
    .def_readonly("minor", &cudaDeviceProp::minor)
    .def_readonly("totalGlobalMem", &cudaDeviceProp::totalGlobalMem)
    .def_readonly("totalConstMem", &cudaDeviceProp::totalConstMem)
    .def_readonly("name", &cudaDeviceProp::name)
    .def_readonly("multiProcessorCount", &cudaDeviceProp::multiProcessorCount)
    .def_readonly("maxThreadsPerMultiProcessor", &cudaDeviceProp::maxThreadsPerMultiProcessor)
    .def_readonly("mangedMemory", &cudaDeviceProp::managedMemory)
    .def_readonly("sharedMemPerBlock", &cudaDeviceProp::sharedMemPerBlock)
    .def_readonly("sharedMemPerBlockOptin", &cudaDeviceProp::sharedMemPerBlockOptin)
    .def_readonly("sharedMemPerMultiprocessor", &cudaDeviceProp::sharedMemPerMultiprocessor)
    .def_readonly("unifiedAddressing", &cudaDeviceProp::unifiedAddressing)
    .def_readonly("unifiedFunctionPointers", &cudaDeviceProp::unifiedFunctionPointers)
    .def_readonly("concurrentKernels", &cudaDeviceProp::concurrentKernels)
    .def_readonly("concurrentManagedAccess", &cudaDeviceProp::concurrentManagedAccess)
    .def_readonly("directManagedMemAccessFromHost", &cudaDeviceProp::directManagedMemAccessFromHost)
    .def_property_readonly("maxTexture3D", [](const cudaDeviceProp& prop){
      return pybind11::make_tuple(
        prop.maxTexture3D[0],
        prop.maxTexture3D[1],
        prop.maxTexture3D[2]
      );
    })
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

  pybind11::class_<cudaError_t>(m, "CudaError")
    .def("__int__", [](const cudaError_t& error){
      return static_cast<int>(error);
    })
    .def("__bool__", [](const cudaError_t& error){
      return error != cudaSuccess;
    })
    .def("__repr__", [](const cudaError_t& error){
      std::stringstream ss;
      ss << cudaGetErrorName(error) << " - " << cudaGetErrorString(error);

      return ss.str();
    });

  pybind11::class_<dicomNodeError_t>(m, "DicomnodeError")
    .def("__repr__", [](const dicomNodeError_t& error){
      if(!error){
        return std::string("Success");
      }
      if(is_cuda_error(error)) {
        std::stringstream ss;
        cudaError_t cuda_error = extract_cuda_error(error);
        ss << "Encoutered cuda error:" << cudaGetErrorName(cuda_error) << " - " << cudaGetErrorString(cuda_error);
        return ss.str();
      }

      return std::string("ERROR, raise this as a value!");
    })
    .def("__int__", [](const dicomNodeError_t& error){
      return static_cast<uint32_t>(error);
    })
    .def("__bool__",[](const dicomNodeError_t& error){
      return error != dicomNodeError_t::SUCCESS;
    });

  m.def("print_image", &print_image);
  m.def("get_device_properties", &cast_current_device);
}
