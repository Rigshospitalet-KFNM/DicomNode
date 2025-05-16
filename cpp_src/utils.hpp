#pragma once
#include<iostream>
#include<pybind11/pybind11.h>
#include"core/core.hpp"

namespace dicomnode {
inline dicomNodeError_t is_instance(
  const pybind11::object& python_object,
  const char* module_name,
  const char* instance_type){
  const pybind11::module_& space_module = pybind11::module_::import(module_name);
  const pybind11::object& space_class = space_module.attr(instance_type);

  if(!pybind11::isinstance(python_object, space_class)) {
    return dicomNodeError_t::INPUT_TYPE_ERROR;
  }
  return dicomNodeError_t::SUCCESS;
}

inline dicomNodeError_t check_buffer_pointers(
  const pybind11::buffer_info& buffer, const size_t elements
){
  if (buffer.ptr == nullptr) {
    return UNABLE_TO_ACQUIRE_BUFFER;
  }

  if (buffer.size != elements){
    std::cerr << "Input Size miss match buffer: " << buffer.size << " Elements:" << elements << "\n";
    return INPUT_SIZE_MISMATCH;
  }

  return dicomNodeError_t::SUCCESS;
}

  /**
 * @brief Returns the amount of bytes spaned by an image object or a space for
 * an image object
 *
 * @tparam T
 * @param python_object
 * @return size_t
 */
template<typename T>
inline size_t get_image_size(const pybind11::object& python_object){
  const pybind11::module_& space_module = pybind11::module_::import("dicomnode.math.space");
  const pybind11::object& space_class = space_module.attr("Space");

  if(pybind11::isinstance(python_object, space_class)){
    using EXTENT_DATA_TYPE = uint32_t;
    const pybind11::array_t<EXTENT_DATA_TYPE>& python_extent = python_object.attr("extent");
    const pybind11::buffer_info& python_extent_buffer = python_extent.request();
    if (python_extent_buffer.ptr == nullptr) {
      return 0;
    }

    EXTENT_DATA_TYPE* data = (EXTENT_DATA_TYPE*)python_extent_buffer.ptr;

    size_t size = sizeof(T);
    for(int i = 0; i < python_extent_buffer.size; i++){
      size *= data[i];
    }

    return size;
  }

  const pybind11::module_& image_module = pybind11::module_::import("dicomnode.math.image");
  const pybind11::object& image_class = image_module.attr("Image");

  if(pybind11::isinstance(python_object, image_class)){
    const pybind11::array_t<T>& raw_image = python_object.attr("raw");
    return raw_image.size() * sizeof(T);
  }

  return 0;
}

template<typename T>
dicomNodeError_t get_python_buffer_pointer(
  const pybind11::object& obj,
  const char* attr,
  const size_t expected_size,
  T** out,
  bool writable=false
){
  const pybind11::array_t<T>& python_array = obj.attr(attr).cast<const pybind11::array_t<T>&>();

  const pybind11::buffer_info buffer = python_array.request(writable);
  if (buffer.ptr == NULL){
    return UNABLE_TO_ACQUIRE_BUFFER;
  }

  if (buffer.size != expected_size){
    return INPUT_SIZE_MISMATCH;
  }

  *out = static_cast<T*>(buffer.ptr);

  return SUCCESS;
}

template<u8 DIM>
dicomNodeError_t load_space(
  const pybind11::object& python_space,
  Space<DIM>& space
){

  DicomNodeRunner runner;

  f32* start_point_ptr = NULL;
  f32* basis_ptr = NULL;
  f32* inverted_basis_ptr = NULL;
  u32* extent_ptr = NULL;

  runner
    | [&](){
      return is_instance(python_space, "dicomnode.math.space", "Space");
  } | [&](){
    return get_python_buffer_pointer(
      python_space,
      space.starting_point_attr_name,
      DIM,
      &start_point_ptr
    );
  } | [&](){
    return get_python_buffer_pointer(
      python_space,
      space.basis_attr_name,
      DIM * DIM,
      &basis_ptr
    );
  } | [&](){
    return get_python_buffer_pointer(
      python_space,
      space.inverted_basis_attr_name,
      DIM * DIM,
      &inverted_basis_ptr
    );
  } | [&](){
    return get_python_buffer_pointer(
      python_space,
      space.extent_attr_name,
      DIM,
      &extent_ptr
    );
  } | [&](){
    std::memcpy(
      space.starting_point.points.begin(),
      start_point_ptr,
      sizeof(f32) * DIM
    );

    std::memcpy(
      space.basis.points.begin(),
      basis_ptr,
      sizeof(f32) * DIM * DIM
    );

    std::memcpy(
      space.inverted_basis.points.begin(),
      start_point_ptr,
      sizeof(f32) * DIM * DIM
    );

    std::memcpy(
      space.extent.begin(),
      extent_ptr,
      sizeof(u32) * DIM
    );

    return SUCCESS;
  };
  return runner.error();
}

} // NAME SPACE END
