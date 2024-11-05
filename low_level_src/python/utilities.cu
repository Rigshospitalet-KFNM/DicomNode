#include"utilities.cuh"

dicomNodeError_t _load_into_host_space(
  Space<3>* host_space,
  const pybind11::object& space
){
  const python_array<float>& starting_point = space.attr("starting_point").cast<python_array<float>>();
  const python_array<float>& basis = space.attr("basis").cast<python_array<float>>();
  const python_array<float>& inv_basis = space.attr("inverted_basis").cast<python_array<float>>();
  const python_array<float>& domain = space.attr("domain").cast<python_array<int>>();

  const pybind11::buffer_info& starting_point_buffer = starting_point.request();
  const pybind11::buffer_info& basis_buffer = basis.request();
  const pybind11::buffer_info& inv_basis_buffer = inv_basis.request();
  const pybind11::buffer_info& domain_buffer = inv_basis.request();

  dicomNodeError_t error = check_buffer_pointers(
    std::cref(basis_buffer), host_space->basis.elements(),
    std::cref(inv_basis_buffer), host_space->inverted_basis.elements(),
    std::cref(starting_point_buffer), host_space->starting_point.elements(),
    std::cref(domain_buffer), host_space->domain.elements()
  );

  if(error){
    return error;
  }

  std::memcpy(&host_space->basis.points, basis_buffer.ptr, host_space->basis.elements() * sizeof(float));
  std::memcpy(&host_space->inverted_basis.points, inv_basis_buffer.ptr, host_space->inverted_basis.elements() * sizeof(float));
  std::memcpy(&host_space->starting_point.points, starting_point_buffer.ptr, host_space->starting_point.elements() * sizeof(float));
  std::memcpy(&host_space->domain.sizes, domain_buffer.ptr, host_space->domain.elements() * sizeof(int));
  // This is just for RVO - the value is SUCCESS
  return error;
}

dicomNodeError_t _load_into_device_space(
  Space<3>* device_space,
  const pybind11::object& space
){
  const python_array<float>& starting_point = space.attr("starting_point").cast<python_array<float>>();
  const python_array<float>& inv_basis = space.attr("inverted_basis").cast<python_array<float>>();
  const python_array<float>& basis = space.attr("basis").cast<python_array<float>>();
  const python_array<float>& domain = space.attr("domain").cast<python_array<int>>();

  const pybind11::buffer_info& starting_point_buffer = starting_point.request();
  const pybind11::buffer_info& basis_buffer = basis.request();
  const pybind11::buffer_info& inv_basis_buffer = inv_basis.request();
  const pybind11::buffer_info& domain_buffer = inv_basis.request();

  DicomNodeRunner runner;
  runner
    | [&](){
      dicomNodeError_t error = check_buffer_pointers(
        std::cref(basis_buffer), device_space->basis.elements(),
        std::cref(inv_basis_buffer), device_space->inverted_basis.elements(),
        std::cref(starting_point_buffer), device_space->starting_point.elements(),
        std::cref(domain_buffer), device_space->domain.elements()
      );

      return error;
    }
    | [&](){ return cudaMemcpy(
      &device_space->basis.points,
      basis_buffer.ptr,
      device_space->basis.elements() * sizeof(float),
      cudaMemcpyDefault);
    }
    | [&](){ return cudaMemcpy(
      &device_space->inverted_basis.points,
      inv_basis_buffer.ptr,
      device_space->inverted_basis.elements() * sizeof(float),
      cudaMemcpyDefault);
    }
    | [&](){ return cudaMemcpy(
      &device_space->starting_point.points,
      starting_point_buffer.ptr,
      device_space->starting_point.elements() * sizeof(float),
      cudaMemcpyDefault);
    }
    | [&](){ return cudaMemcpy(
      &device_space->domain.sizes,
      domain_buffer.ptr,
      device_space->domain.elements() * sizeof(int),
      cudaMemcpyDefault);};

  return runner.error();
}


dicomNodeError_t check_buffer_pointers(
  const pybind11::buffer_info& buffer, const size_t elements
){
  if (buffer.ptr == nullptr) {
    return UNABLE_TO_ACQUIRE_BUFFER;
  }

  if (buffer.size != elements){
    return INPUT_SIZE_MISMATCH;
  }

  return dicomNodeError_t::SUCCESS;
}

std::string get_byte_string (size_t bytes){
  std::stringstream ss;
  if(bytes < 1024){
    ss << bytes << " B";
    return ss.str();
  }
  bytes >>= 10;
  if(bytes < 1024){
    ss << bytes << " kB";
    return ss.str();
  }
  bytes >>= 10;
  if(bytes < 1024){
    ss << bytes << " MB";
    return ss.str();
  }

  bytes >>= 10;
  ss << bytes << " GB";
  return ss.str();
}

dicomNodeError_t load_space(Space<3>* space, const pybind11::object& python_space){
  dicomNodeError_t ret;
  // Note that these can throw, if somebody fuck with the python installation...
  const pybind11::module_& space_module = pybind11::module_::import("dicomnode.math.space");
  const pybind11::object& space_class = space_module.attr("Space");

  if(!pybind11::isinstance(python_space, space_class)){
    ret = dicomNodeError_t::INPUT_TYPE_ERROR;
    return ret;
  }

  cudaPointerAttributes attr;
  cudaError_t error = cudaPointerGetAttributes(&attr, space);
  if(error){
    return encode_cuda_error(error);
  }

  if(attr.type == cudaMemoryType::cudaMemoryTypeUnregistered || attr.type == cudaMemoryType::cudaMemoryTypeHost){
    ret = _load_into_host_space(space, python_space);
  } else {
    ret = _load_into_device_space(space, python_space);
  }

  return ret;
}
