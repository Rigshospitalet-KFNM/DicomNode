#include"utilities.cuh"

dicomNodeError_t _load_into_host_space(
  Space<3>* host_space,
  const pybind11::object& space
){
  if(host_space == nullptr){
    return ARG_IS_NULL_POINTER;
  }

  const python_array<float>& starting_point = space.attr("starting_point").cast<python_array<float>>();
  const python_array<float>& basis = space.attr("basis").cast<python_array<float>>();
  const python_array<float>& inv_basis = space.attr("inverted_basis").cast<python_array<float>>();
  const python_array<uint32_t>& extent = space.attr("extent").cast<python_array<uint32_t>>();

  const pybind11::buffer_info& starting_point_buffer = starting_point.request();
  const pybind11::buffer_info& basis_buffer = basis.request();
  const pybind11::buffer_info& inv_basis_buffer = inv_basis.request();
  const pybind11::buffer_info& extent_buffer = extent.request();

  DicomNodeRunner runner([](const dicomNodeError_t error){
    std::cout << "Load in to host space encountered:" << error_to_human_readable(error);
  });

  runner
    | [&](){
      return check_buffer_pointers(
        std::cref(basis_buffer), host_space->basis.elements()
      );
    } | [&](){

      return check_buffer_pointers(
        std::cref(inv_basis_buffer), host_space->inverted_basis.elements()
      );
    } | [&](){

      return check_buffer_pointers(
        std::cref(starting_point_buffer), host_space->starting_point.elements()
      );
    } | [&](){

      return check_buffer_pointers(
        std::cref(extent_buffer), host_space->extent.dimensionality()
      );
    } | [&](){
      std::memcpy(host_space->basis.points, basis_buffer.ptr, host_space->basis.elements() * sizeof(float));
      std::memcpy(host_space->inverted_basis.points, inv_basis_buffer.ptr, host_space->inverted_basis.elements() * sizeof(float));
      std::memcpy(host_space->starting_point.points, starting_point_buffer.ptr, host_space->starting_point.elements() * sizeof(float));
      std::memcpy(host_space->extent.sizes, extent_buffer.ptr, host_space->extent.dimensionality() * sizeof(uint32_t));

      return dicomNodeError_t::SUCCESS;
    };

  return runner.error();
}

dicomNodeError_t _load_into_device_space(
  Space<3>* device_space,
  const pybind11::object& space
){
  if(device_space == nullptr){
    return ARG_IS_NULL_POINTER;
  }

  const python_array<float>& starting_point = space.attr("starting_point").cast<python_array<float>>();
  const python_array<float>& inv_basis = space.attr("inverted_basis").cast<python_array<float>>();
  const python_array<float>& basis = space.attr("basis").cast<python_array<float>>();
  const python_array<uint32_t>& extent = space.attr("extent").cast<python_array<uint32_t>>();

  const pybind11::buffer_info& starting_point_buffer = starting_point.request();
  const pybind11::buffer_info& basis_buffer = basis.request();
  const pybind11::buffer_info& inv_basis_buffer = inv_basis.request();
  const pybind11::buffer_info& extent_buffer = extent.request();

  DicomNodeRunner runner{
    [](const dicomNodeError_t& error){
      std::cout << "_load_into_device_space encountered error: " << (uint32_t)error << "\n";
    }
  };
  runner
    | [&](){
      return check_buffer_pointers(
        std::cref(basis_buffer), device_space->basis.elements()
      );
    } | [&](){

      return check_buffer_pointers(
        std::cref(inv_basis_buffer), device_space->inverted_basis.elements()
      );
    } | [&](){

      return check_buffer_pointers(
        std::cref(starting_point_buffer), device_space->starting_point.elements()
      );
    } | [&](){

      return check_buffer_pointers(
        std::cref(extent_buffer), device_space->extent.dimensionality()
      );
    } | [&](){ return cudaMemcpy(
      device_space->basis.points,
      basis_buffer.ptr,
      device_space->basis.elements() * sizeof(float),
      cudaMemcpyDefault);
    }
    | [&](){ return cudaMemcpy(
      device_space->inverted_basis.points,
      inv_basis_buffer.ptr,
      device_space->inverted_basis.elements() * sizeof(float),
      cudaMemcpyDefault);
    }
    | [&](){ return cudaMemcpy(
      device_space->starting_point.points,
      starting_point_buffer.ptr,
      device_space->starting_point.elements() * sizeof(float),
      cudaMemcpyDefault);
    }
    | [&](){ return cudaMemcpy(
      device_space->extent.sizes,
      extent_buffer.ptr,
      device_space->extent.dimensionality() * sizeof(uint32_t),
      cudaMemcpyDefault);};

  return runner.error();
}

dicomNodeError_t is_instance(
  const pybind11::object& python_object,
  const char* module_name,
  const char* instance_type){
  const pybind11::module_& space_module = pybind11::module_::import(module_name);
  const pybind11::object& space_class = space_module.attr(instance_type);

  if(!pybind11::isinstance(python_object, space_class)){
    return dicomNodeError_t::INPUT_TYPE_ERROR;
  }
  return dicomNodeError_t::SUCCESS;
}

dicomNodeError_t check_buffer_pointers(
  const pybind11::buffer_info& buffer, const size_t elements
){
  if (buffer.ptr == nullptr) {
    return UNABLE_TO_ACQUIRE_BUFFER;
  }

  if (buffer.size != elements){
    std::cout << "buffer: " << buffer.size << " Elements:" << elements << "\n";
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
  if(space == nullptr){
    return ARG_IS_NULL_POINTER;
  }

  cudaPointerAttributes attr;
  DicomNodeRunner runner{
    [](const dicomNodeError_t& error){
      std::cout << "load_space encountered error: " << (uint32_t)error << "\n";
    }
  };
  runner
    | [&](){
      return is_instance(python_space, "dicomnode.math.space", "Space");
    } | [&](){
      return cudaPointerGetAttributes(&attr, space);
    } | [&](){
      if(attr.type == cudaMemoryType::cudaMemoryTypeUnregistered || attr.type == cudaMemoryType::cudaMemoryTypeHost){
        return _load_into_host_space(space, python_space);
      } else {
        return _load_into_device_space(space, python_space);
      }
    };

  return runner.error();
}
