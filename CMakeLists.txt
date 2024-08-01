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
  pybind11_add_module(_cuda SHARED
                      low_level_src/python_entry_point.cu)

  # In debug create tests:
  if(NOT DEFINED NO_TESTING)
  include(FetchContent)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG e397860
    )
    # For Windows: Prevent overriding the parent project's compiler/linker settings
    #set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
    enable_testing()

    add_executable(cu_tests low_level_src/test_initiator.cu)
    target_link_libraries(
      cu_tests
      GTest::gtest_main
    )
    include(GoogleTest)
    gtest_discover_tests(cu_tests)
    # Sadly Coverage for GPU code is not supported...
    # https://docs.nvidia.com/cuda/cuda-features-archive/index.html#bullseye-support

  endif()

else()
  message(STATUS "No Cuda Compiler!")
endif()
# -----------------------------------------------------------------------
