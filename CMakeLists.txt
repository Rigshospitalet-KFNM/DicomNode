cmake_minimum_required(VERSION 3.24)
project(dicomnode_c LANGUAGES CXX)

include(ProcessorCount)

ProcessorCount(NUMBER_OF_PROCESSORS)

if(NOT NUMBER_OF_PROCESSORS EQUAL 0)
  message(STATUS "Processors ${NUMBER_OF_PROCESSORS}")
  set(CMAKE_BUILD_PARALLEL_LEVEL ${NUMBER_OF_PROCESSORS})
  set_property(GLOBAL PROPERTY JOB_POOLS compile_pool=${NUMBER_OF_PROCESSORS})
  set(CMAKE_JOB_POOL_COMPILE compile_pool)

endif()


set(VERSION_INFO 0.0.1)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)

function(add_py_object_library name source)
  add_library(${name} OBJECT ${source})
  target_link_libraries(${name} PRIVATE pybind11::module Python::Python utilities)
  target_include_directories(${name} PRIVATE ${pybind11_INCLUDE_DIRS} ${Python_INCLUDE_DIRS})
  target_compile_options(${name} PRIVATE -O3)
  set_property(TARGET ${name} PROPERTY POSITION_INDEPENDENT_CODE ON)
  set_property(TARGET ${name} PROPERTY CUDA_ARCHITECTURES native)
endfunction()

# Create object libraries for each source file

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
  message(status ${PYTHON_INCLUDE_DIRS})

  add_library(utilities low_level_src/python/utilities.cu)
  target_link_libraries(utilities PRIVATE pybind11::module Python::Python)
  target_include_directories(utilities PRIVATE ${pybind11_INCLUDE_DIRS} ${Python_INCLUDE_DIRS})

  set(CUDA_LIBRARIES
    bounding_box low_level_src/python/python_bounding_box.cu
    cuda_management low_level_src/python/python_cuda_management.cu
    mirror low_level_src/python/python_mirror.cu
    interpolation low_level_src/python/python_interpolation.cu
  )

  set(OBJECT_TARGETS)
  list(LENGTH CUDA_LIBRARIES LIBRARY_COUNT)
  math(EXPR LIBRARY_COUNT "${LIBRARY_COUNT} - 1")
  foreach(INDEX RANGE 0 ${LIBRARY_COUNT} 2)
    math(EXPR SOURCE_INDEX "${INDEX} + 1")
    list(GET CUDA_LIBRARIES ${INDEX} LIBRARY_NAME)
    list(GET CUDA_LIBRARIES ${SOURCE_INDEX} SOURCE_FILE)
    add_py_object_library(${LIBRARY_NAME} ${SOURCE_FILE})
    list(APPEND OBJECT_TARGETS $<TARGET_OBJECTS:${LIBRARY_NAME}>)
  endforeach()


  pybind11_add_module(_cuda SHARED
                      low_level_src/python_entry_point.cu
                      ${OBJECT_TARGETS}
  )

  target_compile_options(_cuda PRIVATE -O3)
  set_property(TARGET _cuda PROPERTY CUDA_ARCHITECTURES native)


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
