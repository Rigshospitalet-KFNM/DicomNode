#include<string>
#include<cstring>
#include<sstream>

#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>

#include"../gpu_code/dicom_node_gpu.cu"

std::string get_byte_string (size_t bytes);

/**
 * @brief This function extracts the underlying cudaError_t that was triggered.
 * Note that this function returns an correct value if it wasn't triggered by a
 * cudaError_t
 * @param error a dicomNodeError_t that was triggered by an none sucess
 * cudaError_t
 * @return cudaError_t
 */
cudaError_t extract_cuda_error(dicomNodeError_t error);

template<typename T>
dicomNodeError_t load_image(
  Image<3, T>* dev_out_image,
  const pybind11::object& python_image
);

template<typename T>
dicomNodeError_t free_image(Image<3, T> dev_ptr);