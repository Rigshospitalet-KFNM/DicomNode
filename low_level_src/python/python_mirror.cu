#include<stdint.h>

#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>

#include"python_constants.cuh"
#include"../gpu_code/dicom_node_gpu.cuh"

#include"utilities.cuh"

// py::array::c_style | py::array::forcecast ensures that the array is a dense
// C style array, I should consider overloading to F style as it's just indexing
// And the depth is O(1)
template<typename OP, typename T>
  requires Mirrors<OP, T, Domain<3>>
int py_mirror(python_array<T> arr){
  const pybind11::buffer_info& arr_buffer = arr.request(true);
  if(arr_buffer.ndim != 3){
    throw std::runtime_error("Input shape must be 3");
  }

  for(ssize_t size : arr_buffer.shape){
    if(size <= 0){
      throw std::runtime_error("One of the shapes does not have a dimension!");
    }
  }

  const Domain<3> domain(
    arr_buffer.shape[0],
    arr_buffer.shape[1],
    arr_buffer.shape[2]
  );

  cudaError_t error = mirror_in_place<OP, T>((T*)arr_buffer.ptr, domain);

  return (int)error;
}

void apply_mirror_module(pybind11::module& m){
  const char* mirror_x_name = "mirror_x";
  const char* mirror_y_name = "mirror_y";
  const char* mirror_z_name = "mirror_z";
  const char* mirror_xy_name = "mirror_xy";
  const char* mirror_xz_name = "mirror_xz";
  const char* mirror_yz_name = "mirror_yz";
  const char* mirror_xyz_name = "mirror_xyz";

  const char* mirror_x_doc = "Mirror as 3D volume along the X axis";
  const char* mirror_y_doc = "Mirror as 3D volume along the Y axis";
  const char* mirror_z_doc = "Mirror as 3D volume along the Z axis";
  const char* mirror_xy_doc = "Mirror as 3D volume along the X axis and the Y axis";
  const char* mirror_xz_doc = "Mirror as 3D volume along the X axis and the Z axis";
  const char* mirror_yz_doc = "Mirror as 3D volume along the Y axis and the Z axis";
  const char* mirror_xyz_doc = "Mirror as 3D volume along the X,Y,Z axis";

  m.def(mirror_x_name, &py_mirror<Mirror_X<double>, double>, mirror_x_doc);
  m.def(mirror_x_name, &py_mirror<Mirror_X<float>, float>, mirror_x_doc);
  m.def(mirror_x_name, &py_mirror<Mirror_X<int8_t>, int8_t>, mirror_x_doc);
  m.def(mirror_x_name, &py_mirror<Mirror_X<int16_t>, int16_t>, mirror_x_doc);
  m.def(mirror_x_name, &py_mirror<Mirror_X<int32_t>, int32_t>, mirror_x_doc);
  m.def(mirror_x_name, &py_mirror<Mirror_X<int64_t>, int64_t>, mirror_x_doc);
  m.def(mirror_x_name, &py_mirror<Mirror_X<uint8_t>, uint8_t>, mirror_x_doc);
  m.def(mirror_x_name, &py_mirror<Mirror_X<uint16_t>, uint16_t>, mirror_x_doc);
  m.def(mirror_x_name, &py_mirror<Mirror_X<uint32_t>, uint32_t>, mirror_x_doc);
  m.def(mirror_x_name, &py_mirror<Mirror_X<uint64_t>, uint64_t>, mirror_x_doc);


  m.def(mirror_y_name, &py_mirror<Mirror_Y<double>, double>, mirror_y_doc);
  m.def(mirror_y_name, &py_mirror<Mirror_Y<float>, float>, mirror_y_doc);
  m.def(mirror_y_name, &py_mirror<Mirror_Y<int8_t>, int8_t>, mirror_y_doc);
  m.def(mirror_y_name, &py_mirror<Mirror_Y<int16_t>, int16_t>, mirror_y_doc);
  m.def(mirror_y_name, &py_mirror<Mirror_Y<int32_t>, int32_t>, mirror_y_doc);
  m.def(mirror_y_name, &py_mirror<Mirror_Y<int64_t>, int64_t>, mirror_y_doc);
  m.def(mirror_y_name, &py_mirror<Mirror_Y<uint8_t>, uint8_t>, mirror_y_doc);
  m.def(mirror_y_name, &py_mirror<Mirror_Y<uint16_t>, uint16_t>, mirror_y_doc);
  m.def(mirror_y_name, &py_mirror<Mirror_Y<uint32_t>, uint32_t>, mirror_y_doc);
  m.def(mirror_y_name, &py_mirror<Mirror_Y<uint64_t>, uint64_t>, mirror_y_doc);

  m.def(mirror_z_name, &py_mirror<Mirror_Z<double>, double>, mirror_z_doc);
  m.def(mirror_z_name, &py_mirror<Mirror_Z<float>, float>, mirror_z_doc);
  m.def(mirror_z_name, &py_mirror<Mirror_Z<int8_t>, int8_t>, mirror_z_doc);
  m.def(mirror_z_name, &py_mirror<Mirror_Z<int16_t>, int16_t>, mirror_z_doc);
  m.def(mirror_z_name, &py_mirror<Mirror_Z<int32_t>, int32_t>, mirror_z_doc);
  m.def(mirror_z_name, &py_mirror<Mirror_Z<int64_t>, int64_t>, mirror_z_doc);
  m.def(mirror_z_name, &py_mirror<Mirror_Z<uint8_t>, uint8_t>, mirror_z_doc);
  m.def(mirror_z_name, &py_mirror<Mirror_Z<uint16_t>, uint16_t>, mirror_z_doc);
  m.def(mirror_z_name, &py_mirror<Mirror_Z<uint32_t>, uint32_t>, mirror_z_doc);
  m.def(mirror_z_name, &py_mirror<Mirror_Z<uint64_t>, uint64_t>, mirror_z_doc);

  m.def(mirror_xy_name, &py_mirror<Mirror_XY<double>, double>, mirror_xy_doc);
  m.def(mirror_xy_name, &py_mirror<Mirror_XY<float>, float>, mirror_xy_doc);
  m.def(mirror_xy_name, &py_mirror<Mirror_XY<int8_t>, int8_t>, mirror_xy_doc);
  m.def(mirror_xy_name, &py_mirror<Mirror_XY<int16_t>, int16_t>, mirror_xy_doc);
  m.def(mirror_xy_name, &py_mirror<Mirror_XY<int32_t>, int32_t>, mirror_xy_doc);
  m.def(mirror_xy_name, &py_mirror<Mirror_XY<int64_t>, int64_t>, mirror_xy_doc);
  m.def(mirror_xy_name, &py_mirror<Mirror_XY<uint8_t>, uint8_t>, mirror_xy_doc);
  m.def(mirror_xy_name, &py_mirror<Mirror_XY<uint16_t>, uint16_t>, mirror_xy_doc);
  m.def(mirror_xy_name, &py_mirror<Mirror_XY<uint32_t>, uint32_t>, mirror_xy_doc);
  m.def(mirror_xy_name, &py_mirror<Mirror_XY<uint64_t>, uint64_t>, mirror_xy_doc);

  m.def(mirror_xz_name, &py_mirror<Mirror_XZ<double>, double>, mirror_xz_doc);
  m.def(mirror_xz_name, &py_mirror<Mirror_XZ<float>, float>, mirror_xz_doc);
  m.def(mirror_xz_name, &py_mirror<Mirror_XZ<int8_t>, int8_t>, mirror_xz_doc);
  m.def(mirror_xz_name, &py_mirror<Mirror_XZ<int16_t>, int16_t>, mirror_xz_doc);
  m.def(mirror_xz_name, &py_mirror<Mirror_XZ<int32_t>, int32_t>, mirror_xz_doc);
  m.def(mirror_xz_name, &py_mirror<Mirror_XZ<int64_t>, int64_t>, mirror_xz_doc);
  m.def(mirror_xz_name, &py_mirror<Mirror_XZ<uint8_t>, uint8_t>, mirror_xz_doc);
  m.def(mirror_xz_name, &py_mirror<Mirror_XZ<uint16_t>, uint16_t>, mirror_xz_doc);
  m.def(mirror_xz_name, &py_mirror<Mirror_XZ<uint32_t>, uint32_t>, mirror_xz_doc);
  m.def(mirror_xz_name, &py_mirror<Mirror_XZ<uint64_t>, uint64_t>, mirror_xz_doc);

  m.def(mirror_yz_name, &py_mirror<Mirror_YZ<double>, double>, mirror_yz_doc);
  m.def(mirror_yz_name, &py_mirror<Mirror_YZ<float>, float>, mirror_yz_doc);
  m.def(mirror_yz_name, &py_mirror<Mirror_YZ<int8_t>, int8_t>, mirror_yz_doc);
  m.def(mirror_yz_name, &py_mirror<Mirror_YZ<int16_t>, int16_t>, mirror_yz_doc);
  m.def(mirror_yz_name, &py_mirror<Mirror_YZ<int32_t>, int32_t>, mirror_yz_doc);
  m.def(mirror_yz_name, &py_mirror<Mirror_YZ<int64_t>, int64_t>, mirror_yz_doc);
  m.def(mirror_yz_name, &py_mirror<Mirror_YZ<uint8_t>, uint8_t>, mirror_yz_doc);
  m.def(mirror_yz_name, &py_mirror<Mirror_YZ<uint16_t>, uint16_t>, mirror_yz_doc);
  m.def(mirror_yz_name, &py_mirror<Mirror_YZ<uint32_t>, uint32_t>, mirror_yz_doc);
  m.def(mirror_yz_name, &py_mirror<Mirror_YZ<uint64_t>, uint64_t>, mirror_yz_doc);

  m.def(mirror_xyz_name, &py_mirror<Mirror_XYZ<double>, double>, mirror_xyz_doc);
  m.def(mirror_xyz_name, &py_mirror<Mirror_XYZ<float>, float>, mirror_xyz_doc);
  m.def(mirror_xyz_name, &py_mirror<Mirror_XYZ<int8_t>, int8_t>, mirror_xyz_doc);
  m.def(mirror_xyz_name, &py_mirror<Mirror_XYZ<int16_t>, int16_t>, mirror_xyz_doc);
  m.def(mirror_xyz_name, &py_mirror<Mirror_XYZ<int32_t>, int32_t>, mirror_xyz_doc);
  m.def(mirror_xyz_name, &py_mirror<Mirror_XYZ<int64_t>, int64_t>, mirror_xyz_doc);
  m.def(mirror_xyz_name, &py_mirror<Mirror_XYZ<uint8_t>, uint8_t>, mirror_xyz_doc);
  m.def(mirror_xyz_name, &py_mirror<Mirror_XYZ<uint16_t>, uint16_t>, mirror_xyz_doc);
  m.def(mirror_xyz_name, &py_mirror<Mirror_XYZ<uint32_t>, uint32_t>, mirror_xyz_doc);
  m.def(mirror_xyz_name, &py_mirror<Mirror_XYZ<uint64_t>, uint64_t>, mirror_xyz_doc);

}