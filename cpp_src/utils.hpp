#pragma once

#include<iostream>

#include<pybind11/pybind11.h>

#include"core/core.hpp"

namespace dicomnode {
inline CppError_t is_instance(
  const pybind11::object& python_object,
  const char* module_name,
  const char* instance_type){
  const pybind11::module_& module_ = pybind11::module_::import(module_name);
  const pybind11::object& class_ = module_.attr(instance_type);

  if(!pybind11::isinstance(python_object, class_)) {
    return CppError_t::INPUT_TYPE_ERROR;
  }
  return CppError_t::SUCCESS;
}

inline CppError_t check_buffer_pointers(
  const pybind11::buffer_info& buffer, const size_t elements
){
  if (buffer.ptr == nullptr) {
    return UNABLE_TO_ACQUIRE_BUFFER;
  }

  if (std::cmp_not_equal(buffer.size, elements)){
    std::cerr << "Input Size miss match buffer: " << buffer.size << " Elements:" << elements << "\n";
    return INPUT_SIZE_MISMATCH;
  }

  return CppError_t::SUCCESS;
}


template<typename T>
CppError_t get_python_buffer_pointer(
  const pybind11::object& obj,
  const char* attr,
  const size_t expected_size,
  T** out,
  bool writable=false
){
  const pybind11::array_t<T>& python_array = obj.attr(attr).cast<const pybind11::array_t<T>>();

  const pybind11::buffer_info buffer = python_array.request(writable);
  if (buffer.ptr == NULL){
    return UNABLE_TO_ACQUIRE_BUFFER;
  }

  if (buffer.size == SSIZE_ERROR || static_cast<u64>(buffer.size) != expected_size){
    return INPUT_SIZE_MISMATCH;
  }

  *out = static_cast<T*>(buffer.ptr);

  return SUCCESS;
}

template<u8 DIM>
CppError_t load_space(
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
    if(  start_point_ptr == NULL
      || basis_ptr == NULL
      || inverted_basis_ptr == NULL
      || extent_ptr == NULL
    ) {
      std::cout << "start_point_ptr:" << start_point_ptr << "\n"
                << "basis_ptr:" << basis_ptr << "\n"
                << "inverted_basis_ptr:" << inverted_basis_ptr << "\n"
                << "extent_ptr:" << extent_ptr << "\n";

      return INPUT_TYPE_ERROR;
    }

    std::memcpy(
      space.starting_point.begin(),
      start_point_ptr,
      sizeof(f32) * DIM
    );

    std::memcpy(
      space.basis.begin(),
      basis_ptr,
      sizeof(f32) * DIM * DIM
    );

    std::memcpy(
      space.inverted_basis.begin(),
      inverted_basis_ptr,
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
