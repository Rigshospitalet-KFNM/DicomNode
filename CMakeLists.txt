cmake_minimum_required(VERSION 3.18)
project(dicomnode_c LANGUAGES CXX)

set(VERSION_INFO 0.0.1)

find_package(Python COMPONENTS Interpreter Development)

if(EXISTS  "${CMAKE_SOURCE_DIR}/extern/pybind11")
  add_subdirectory(extern/pybind11 REQUIRED)
else()
  find_package(pybind11 CONFIG REQUIRED)
endif()


pybind11_add_module(_c low_level_src/low_level.cpp)
target_compile_definitions(_c
                           PRIVATE VERSION_INFO={EXAMPLE_VERSION_INFO})
#set_target_properties(_c PROPERTIES OUTPUT_NAME dicomnode._c)


# -----------------------------------------------------------------------
# Checking for CUDA
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  message(STATUS "CUDA enabled!")
  enable_language(CUDA)
  pybind11_add_module(_cuda low_level_src/low_level_cuda.cu)
  target_compile_definitions(_cuda
                             PRIVATE VERSION_INFO={EXAMPLE_VERSION_INFO})
  #set_target_properties(_cuda PROPERTIES OUTPUT_NAME dicomnode._cuda)
else()
  message(STATUS "No Cuda Compiler!")
endif()
# -----------------------------------------------------------------------
