#include"python_registration.cuh"

#include<iostream>
#include<tuple>

#include"../gpu_code/dicom_node_gpu.cuh"
#include"utilities.cuh"



std::tuple<dicomNodeError_t, python_array<f32>> registration(
  pybind11::object& destination_image,
  pybind11::object& source_image
){

  python_array<f32> out;

  Image<3, f32>* gpu_source_image = nullptr;
  Image<3, f32>* gpu_destination_image = nullptr;

  DicomNodeRunner runner([&](dicomNodeError_t error){
    std::cout << "While performing a registration the function encountered error code:" << (u32)error << "\n" ;

    free_device_memory(
      &gpu_source_image,
      &gpu_destination_image
    );
  });

  runner
      | [&](){
      return is_instance(destination_image, "dicomnode.math.image", "Image");
    } | [&](){
      return is_instance(source_image, "dicomnode.math.image", "Image");
    } | [&](){
      return cudaMalloc(&gpu_source_image, sizeof(Image<3, f32>));
    } | [&](){
      return cudaMalloc(&gpu_destination_image, sizeof(Image<3, f32>));
    } | [&](){
      return load_image<f32>(gpu_source_image, source_image);
    } | [&]() {
      return load_image<f32>(gpu_destination_image, destination_image);
    } | [&](){
      return REGISTRATION::register_to(gpu_source_image, gpu_destination_image);
    } | [&](){
      return free_image(gpu_source_image);
    } | [&](){
      return free_image(gpu_destination_image);
    };

  return {runner.error(), out};
}


void apply_registration_module(pybind11::module& m){
  const char* help_message = R"(
    This function registers a destination image to a source image
  )";

  pybind11::module registration_submodule = m.def_submodule(
    "registration"
  );

  registration_submodule.def("register", registration, help_message);
}