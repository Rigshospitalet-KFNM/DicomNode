#pragma once

#include<string>
#include<cstring>
#include<sstream>
#include<tuple>
#include<vector>

#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>

#include"../gpu_code/dicom_node_gpu.cu"

constexpr int PYBIND_ARRAY_FLAGS = pybind11::array::c_style | pybind11::array::forcecast;

template<typename T>
using python_array = pybind11::array_t<T, PYBIND_ARRAY_FLAGS>;


dicomNodeError_t check_buffer_pointers(
  const std::tuple<const pybind11::buffer_info&,const size_t> t
){
  const pybind11::buffer_info& buffer = std::get<0>(t);
  const size_t& elements = std::get<1>(t);
  if (buffer.ptr == nullptr) {
    return UNABLE_TO_AQUIRE_BUFFER;
  }

  if (buffer.size != elements){
    return INPUT_SIZE_MISS_MATCH;
  }

  return dicomNodeError_t::SUCCESS;
}


template<typename... Ts>
dicomNodeError_t check_buffer_pointers(
  const std::tuple<const pybind11::buffer_info&, const size_t> first,
  const Ts...  rest
){
  dicomNodeError_t ret = check_buffer_pointers(first);
  if(ret){
    return ret;
  }

  ret = check_buffer_pointers(rest...);
  return ret;
}


namespace {
  dicomNodeError_t _load_into_host_space(
    Space<3>* host_space,
    const pybind11::object& space
  ){
    const python_array<float>& starting_point = space.attr("starting_point").cast<python_array<float>>();
    const pybind11::buffer_info& starting_point_buffer = starting_point.request();
    const python_array<float>& basis = space.attr("basis").cast<python_array<float>>();
    const pybind11::buffer_info& basis_buffer = basis.request();
    const python_array<float>& inv_basis = space.attr("inverted_basis").cast<python_array<float>>();
    const pybind11::buffer_info& inv_basis_buffer = inv_basis.request();
    const python_array<float>& domain = space.attr("domain").cast<python_array<int>>();
    const pybind11::buffer_info& domain_buffer = inv_basis.request();

    dicomNodeError_t error = check_buffer_pointers(
      std::make_tuple(basis_buffer, host_space->basis.elements()),
      std::make_tuple(inv_basis_buffer, host_space->inverted_basis.elements()),
      std::make_tuple(starting_point_buffer, host_space->starting_point.elements()),
      std::make_tuple(domain_buffer, host_space->domain.elements())
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
    const pybind11::buffer_info& starting_point_buffer = starting_point.request();
    const python_array<float>& basis = space.attr("basis").cast<python_array<float>>();
    const pybind11::buffer_info& basis_buffer = basis.request();
    const python_array<float>& inv_basis = space.attr("inverted_basis").cast<python_array<float>>();
    const pybind11::buffer_info& inv_basis_buffer = inv_basis.request();
    const python_array<float>& domain = space.attr("domain").cast<python_array<int>>();
    const pybind11::buffer_info& domain_buffer = inv_basis.request();

    DicomNodeRunner runner;
    runner
      | [&](){
        dicomNodeError_t error = check_buffer_pointers(
          std::make_tuple(basis_buffer, device_space->basis.elements()),
          std::make_tuple(inv_basis_buffer, device_space->inverted_basis.elements()),
          std::make_tuple(starting_point_buffer, device_space->starting_point.elements()),
          std::make_tuple(domain_buffer, device_space->domain.elements())
        );

        return error;
      }
      | [&](){ return cudaMemcpy(&device_space->basis.points, basis_buffer.ptr, basis_buffer.size * sizeof(float), cudaMemcpyDefault);}
      | [&](){ return cudaMemcpy(&device_space->inverted_basis.points, inv_basis_buffer.ptr, inv_basis_buffer.size * sizeof(float), cudaMemcpyDefault);}
      | [&](){ return cudaMemcpy(&device_space->starting_point.points, starting_point_buffer.ptr, starting_point_buffer.size * sizeof(float), cudaMemcpyDefault);}
      | [&](){ return cudaMemcpy(&device_space->domain.sizes, domain_buffer.ptr, domain_buffer.size * sizeof(int), cudaMemcpyDefault);};

    return runner.error();
  }



  template<typename T>
  dicomNodeError_t _load_into_host_image(
    Image<3, T>* host_out_image,
    const pybind11::object& python_image
  ){
    const python_array<T>& raw_image = python_image.attr("raw").cast<python_array<T>>();
    const pybind11::buffer_info& image_buffer = raw_image.request();
    const pybind11::object& space = python_image.attr("space");
    dicomNodeError_t error = _load_into_host_space(&host_out_image->space, space);
    if(error){
      return error;
    }

    size_t image_size = sizeof(T);
    for (int i = 0; const ssize_t dim : image_buffer.shape){
      if(dim <= 0){
        error = dicomNodeError_t::NON_POSITIVE_SHAPE;
        return error;
      }
      image_size *= dim;
      i++;
    }

    host_out_image->data = new T[image_buffer.size];
    std::memcpy(host_out_image->data, image_buffer.ptr, sizeof(T) * image_buffer.size);

    return error;
  }

  template<typename T>
  dicomNodeError_t _load_into_dev_image(
    Image<3, T>* dev_out_image,
    const pybind11::object& python_image
  ){
    size_t image_size = sizeof(T);
    const python_array<T>& raw_image = python_image.attr("raw").cast<python_array<T>>();
    const pybind11::buffer_info& image_buffer = raw_image.request();
    const pybind11::object& space = python_image.attr("space");

    DicomNodeRunner runner{[&](dicomNodeError_t error){free_device_memory(&dev_out_image->data);}};
    runner
      | [&](){
        return _load_into_dev_space(&(dev_out_image->space), space);
      } | [&](){
        for (const ssize_t& dim : image_buffer.shape){
          if(dim <= 0){
            return dicomNodeError_t::NON_POSITIVE_SHAPE;
          }
          image_size *= dim;
        }
        return dicomNodeError_t::SUCCESS;
      } | [&](){
       return cudaMalloc(&dev_out_image->data, image_size);
      } | [&](){
        return cudaMemcpy(dev_out_image->data, image_buffer.ptr, image_size, cudaMemcpyDefault);
      } | [&](){
      };

    return runner.error();
  }
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


/**
 * @brief This function extracts the underlying cudaError_t that was triggered.
 * Note that this function returns an correct value if it wasn't triggered by a
 * cudaError_t
 * @param error a dicomNodeError_t that was triggered by an none sucess
 * cudaError_t
 * @return cudaError_t
 */
template<typename T>
dicomNodeError_t load_image(Image<3, T>* image, const pybind11::object& python_image){
  // RVO
  dicomNodeError_t ret;
  // Note that these can throw, if somebody fuck with the python installation...
  const pybind11::module_& image_module = pybind11::module_::import("dicomnode.math.image");
  const pybind11::object& image_class = image_module.attr("Image");

  if(!pybind11::isinstance(python_image, image_class)){
    ret = dicomNodeError_t::INPUT_TYPE_ERROR;
    return ret;
  }

  cudaPointerAttributes attr;
  cudaError_t error = cudaPointerGetAttributes(&attr, image);
  if(error){
    return encode_cuda_error(error);
  }

  if(attr.type == cudaMemoryType::cudaMemoryTypeUnregistered || attr.type == cudaMemoryType::cudaMemoryTypeHost){
    ret = _load_into_host_image(image, python_image);
  } else {
    ret = _load_into_dev_image(image, python_image);
  }

  return ret;
}

template<typename T>
size_t get_image_size(const pybind11::object& python_image){
  const python_array<T>& raw_image = python_image.attr("raw").cast<python_array<T>>();
  return raw_image.size() * sizeof(T);
}


template<typename T>
void free_image(Image<3, T>* dev_out_image){
  T* data_pointer = nullptr;
  // Note that if this shit fucks up, we
  cudaMemcpy(&data_pointer, &dev_out_image->data, sizeof(T*), cudaMemcpyDefault);

  cudaFree(data_pointer);
  cudaFree(dev_out_image);
}

cudaError_t extract_cuda_error(dicomNodeError_t error);

std::string get_byte_string (size_t bytes);