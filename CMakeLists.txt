cmake_minimum_required(VERSION 3.22)
project(dicomnode_c LANGUAGES CXX)

set(VERSION_INFO 0.0.1)

add_subdirectory(extern/pybind11 REQUIRED)
pybind11_add_module(dicomnode_c low_level_src/low_level.cpp)
target_compile_definitions(dicomnode_c
                           PRIVATE VERSION_INFO={EXAMPLE_VERSION_INFO})

# -----------------------------------------------------------------------
# Checking for CUDA
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  pybind11_add_module(dicomnode_c_cuda low_level_src/low_level_cuda.cu)
  target_compile_definitions(dicomnode_c_cuda
                            PRIVATE VERSION_INFO={EXAMPLE_VERSION_INFO})
endif()
# -----------------------------------------------------------------------
