#include<string>
#include<cstring>
#include<sstream>

#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>

#include"../gpu_code/dicom_node_gpu.cu"

constexpr int PYBIND_ARRAY_FLAGS = pybind11::array::c_style | pybind11::array::forcecast;

template<typename T>
using python_array = pybind11::array_t<T, PYBIND_ARRAY_FLAGS>;

namespace {
  template<typename T>
  dicomNodeError_t _load_into_host_image(
    Image<3, T>* host_out_image,
    const pybind11::object& python_image
  ){
    const python_array<T> raw_image = python_image.attr("raw").cast<python_array<T>>();
    const pybind11::buffer_info& image_buffer = raw_image.request();
    const pybind11::object& space = python_image.attr("space");
    const python_array<float> starting_point = space.attr("starting_point").cast<python_array<float>>();
    const pybind11::buffer_info& starting_point_buffer = starting_point.request();
    const python_array<float> basis = space.attr("basis").cast<python_array<float>>();
    const pybind11::buffer_info& basis_buffer = basis.request();
    const python_array<float> inv_basis = space.attr("inverted_basis").cast<python_array<float>>();
    const pybind11::buffer_info& inv_basis_buffer = inv_basis.request();

    size_t image_size = sizeof(T);
    for (int i = 0; const ssize_t dim : image_buffer.shape){
      if(dim <= 0){
        return dicomNodeError_t::NON_POSITIVE_SHAPE;
      }
      host_out_image->domain[i] = dim;
      image_size *= dim;
      i++;
    }

    if(image_buffer.ptr == nullptr
      || basis_buffer.ptr == nullptr
      || inv_basis_buffer.ptr == nullptr
      || starting_point_buffer.ptr == nullptr
    ){
      return dicomNodeError_t::UNABLE_TO_AQUIRE_BUFFER;
    }


    host_out_image->data = new T[image_buffer.size];
    std::memcpy(&host_out_image->basis, basis_buffer.ptr, basis_buffer.size * sizeof(float));
    std::memcpy(&host_out_image->inverted_basis, inv_basis_buffer.ptr, inv_basis_buffer.size * sizeof(float));
    std::memcpy(&host_out_image->starting_point, basis_buffer.ptr, basis_buffer.size * sizeof(float));
    std::memcpy(&host_out_image->data, image_buffer.ptr, sizeof(T) * image_buffer.size);

    return dicomNodeError_t::SUCCESS;
  }

  template<typename T>
  dicomNodeError_t _load_into_dev_image(
    Image<3, T>* dev_out_image,
    const pybind11::object& python_image
  ){
    const python_array<T>& raw_image = python_image.attr("raw").cast<python_array<T>>();
    const pybind11::buffer_info& image_buffer = raw_image.request();
    const pybind11::object& space = python_image.attr("space");
    const python_array<float>& starting_point = space.attr("starting_point").cast<python_array<float>>();
    const pybind11::buffer_info& starting_point_buffer = starting_point.request();
    const python_array<float>& basis = space.attr("basis").cast<python_array<float>>();
    const pybind11::buffer_info& basis_buffer = basis.request();
    const python_array<float>& inv_basis = space.attr("inverted_basis").cast<python_array<float>>();
    const pybind11::buffer_info& inv_basis_buffer = inv_basis.request();

    Image<3, T> host_image;

    size_t image_size = sizeof(T);
    for (int i = 0; const ssize_t dim : image_buffer.shape){
      if(dim <= 0){
        return dicomNodeError_t::NON_POSITIVE_SHAPE;
      }
      host_image.domain[i] = dim;
      image_size *= dim;
      i++;
    }

    if(image_buffer.ptr == nullptr
        || basis_buffer.ptr == nullptr
        || inv_basis_buffer.ptr == nullptr
        || starting_point_buffer.ptr) {
          return dicomNodeError_t::UNABLE_TO_AQUIRE_BUFFER;
        }

    std::memcpy(&host_image.basis, basis_buffer.ptr, basis_buffer.size * sizeof(float));
    std::memcpy(&host_image.inverted_basis, inv_basis_buffer.ptr, inv_basis_buffer.size * sizeof(float));
    std::memcpy(&host_image.starting_point, basis_buffer.ptr, basis_buffer.size * sizeof(float));

    DicomNodeRunner runner{[&](dicomNodeError_t error){free_device_memory(&host_image.data);}};

    runner
      | [&](){ return cudaMalloc(&host_image.data, image_size); }
      | [&](){ return cudaMemcpy(host_image.data, image_buffer.ptr, image_size, cudaMemcpyDefault); }
      | [&](){
        return cudaMemcpy(dev_out_image, &host_image, sizeof(Image<3, T>), cudaMemcpyDefault);
      };

    return runner.error();
  }
} // End of anonymous name space

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
void free_image(Image<3, T>* dev_out_image){
  T* data_pointer = nullptr;
  // Note that if this shit fucks up, we
  cudaMemcpy(&data_pointer, &dev_out_image->data, sizeof(T*), cudaMemcpyDefault);

  cudaFree(data_pointer);
  cudaFree(dev_out_image);
}

cudaError_t extract_cuda_error(dicomNodeError_t error);

std::string get_byte_string (size_t bytes);