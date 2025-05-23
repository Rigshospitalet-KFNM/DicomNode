#include"python_constants.cuh"
#include"../gpu_code/dicom_node_gpu.cuh"
#include"utilities.cuh"

#include<tuple>

template<typename T,  uint8_t CHUNK>
std::tuple<cudaError_t, pybind11::list> bounding_box(python_array<T>& arr){
  const pybind11::buffer_info& buffer = arr.request(false);
  if (buffer.ndim != 3){
    throw std::runtime_error("This function requires 3 dimensional input!");
  }

  const ssize_t items = buffer.size;
  if(items < 1){
    throw std::runtime_error("Invalid number of values in buffer");
  }

  const size_t buffer_size = items;
  BoundingBox_3D out;
  Extent<3> space(
    buffer.shape[2], buffer.shape[1], buffer.shape[0]
  );

  cudaError_t error = reduce<1, BoundingBoxOP_3D<T>, T, BoundingBox_3D, Extent<3>>(
    (T*)buffer.ptr,
    buffer_size,
    &out,
    space
  );

  pybind11::list returnList(6);
  returnList[0] = out.x_min;
  returnList[1] = out.x_max;
  returnList[2] = out.y_min;
  returnList[3] = out.y_max;
  returnList[4] = out.z_min;
  returnList[5] = out.z_max;
  return {error, returnList};
}

void apply_bounding_box_module(pybind11::module& m){
  m.def("bounding_box", &bounding_box<float, 1>);
  m.def("bounding_box", &bounding_box<double, 1>);
  m.def("bounding_box", &bounding_box<int8_t, 1>);
  m.def("bounding_box", &bounding_box<int16_t, 1>);
  m.def("bounding_box", &bounding_box<int32_t, 1>);
  m.def("bounding_box", &bounding_box<int64_t, 1>);
  m.def("bounding_box", &bounding_box<uint8_t, 1>);
  m.def("bounding_box", &bounding_box<uint16_t, 1>);
  m.def("bounding_box", &bounding_box<uint32_t, 1>);
  m.def("bounding_box", &bounding_box<uint64_t, 1>);
  m.def("bounding_box", &bounding_box<bool, 1>);
}
