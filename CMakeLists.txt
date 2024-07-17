cmake_minimum_required(VERSION 3.18)
project(dicomnode_c LANGUAGES CXX)


set(VERSION_INFO 0.0.1)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python COMPONENTS Interpreter Development)

find_package(pybind11 REQUIRED)

# -----------------------------------------------------------------------
# Checking for CUDA
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  message(STATUS "CUDA enabled!")
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
#
  pybind11_add_module(_cuda low_level_src/low_level_cuda.cu)
  target_compile_definitions(_cuda
                             PRIVATE VERSION_INFO={EXAMPLE_VERSION_INFO})

else()
  message(STATUS "No Cuda Compiler!")
endif()
# -----------------------------------------------------------------------
