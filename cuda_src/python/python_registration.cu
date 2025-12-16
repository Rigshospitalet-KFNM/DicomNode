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

  Texture<3, f32>* destination_texture = nullptr;
  Texture<3, f32>* source_texture = nullptr;
  DicomNodeRunner runner([&](dicomNodeError_t error){
    std::cout << "While performing a registration the function encountered error code:" << (u32)error << "\n" ;

    free_device_memory(
      &destination_texture,
      &source_texture
    );
  });

  runner
      | [&](){
      return is_instance(destination_image, "dicomnode.math.image", "Image");
    } | [&](){
      return is_instance(source_image, "dicomnode.math.image", "Image");
    } | [&](){
      return cudaMalloc(&destination_texture, sizeof(Texture<3, f32>));
    } | [&](){
      return cudaMalloc(&source_texture, sizeof(Texture<3, f32>));
    } | [&](){
      return load_texture_from_python_image<f32>(destination_texture, destination_image);
    } | [&](){
      return load_texture_from_python_image<f32>(source_texture, source_image);
    } | [&](){
      return free_texture(&destination_texture);
    } | [&](){
      return free_texture(&source_texture);
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