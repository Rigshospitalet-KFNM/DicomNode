cmake_minimum_required(VERSION 3.24)
project(dicomnode_c LANGUAGES CXX)


message("Building with ${CMAKE_VERSION}")

if(CMAKE_VERSION GREATER_EQUAL "3.28")
  SET(CMAKE_CXX_SCAN_FOR_MODULES FALSE)
endif()

include(ProcessorCount)

set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)
set(CMAKE_BUILD_TYPE DEBUG)
set(CMAKE_INSTALL_DO_STRIP FALSE)

ProcessorCount(NUMBER_OF_PROCESSORS)

if(NOT NUMBER_OF_PROCESSORS EQUAL 0)
  message(STATUS "Processors ${NUMBER_OF_PROCESSORS}")
  set(CMAKE_BUILD_PARALLEL_LEVEL ${NUMBER_OF_PROCESSORS})
  set_property(GLOBAL PROPERTY JOB_POOLS compile_pool=${NUMBER_OF_PROCESSORS})
  set(CMAKE_JOB_POOL_COMPILE compile_pool)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
endif()

set(VERSION_INFO 0.0.1)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


include(FetchContent)
if(NOT EXISTS "${CMAKE_CURRENT_BINARY_DIR}/cmake/CPM.cmake")
  message(STATUS "Downloading CPM.cmake")
  file(DOWNLOAD
    https://github.com/cpm-cmake/CPM.cmake/releases/download/v0.42.0/CPM.cmake
    "${CMAKE_CURRENT_BINARY_DIR}/cmake/CPM.cmake"
  )
endif()
include("${CMAKE_CURRENT_BINARY_DIR}/cmake/CPM.cmake")

CPMAddPackage(
  NAME pybind11
  GITHUB_REPOSITORY pybind/pybind11
  GIT_TAG v2.13.6
)

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)

function(add_cpp_library name source)
  add_library(${name} OBJECT ${source})
  target_link_libraries(${name} PRIVATE pybind11::module Python::Python)
  target_precompile_headers(
    ${name}
    PRIVATE
    <pybind11/pybind11.h>
    <pybind11/numpy.h>
  )
  target_include_directories(${name} PRIVATE ${pybind11_INCLUDE_DIRS} ${Python_INCLUDE_DIRS})
  target_compile_options(${name} PRIVATE -O3 -Wall -Wextra)
  set_property(TARGET ${name} PROPERTY POSITION_INDEPENDENT_CODE ON)
endfunction()

set(CPP_LIBRARIES
  cpp_interpolation python_interpolation.cpp
)

set(CPP_OBJECT_TARGETS)
list(LENGTH CPP_LIBRARIES CPP_LIBRARY_COUNT)
math(EXPR CPP_LIBRARY_COUNT "${CPP_LIBRARY_COUNT} - 1")
foreach(INDEX RANGE 0 ${CPP_LIBRARY_COUNT} 2)
  math(EXPR SOURCE_INDEX "${INDEX} + 1")
  list(GET CPP_LIBRARIES ${INDEX} LIBRARY_NAME)
  list(GET CPP_LIBRARIES ${SOURCE_INDEX} SOURCE_FILE)
  add_cpp_library(${LIBRARY_NAME} ${SOURCE_FILE})
  list(APPEND CPP_OBJECT_TARGETS $<TARGET_OBJECTS:${LIBRARY_NAME}>)
endforeach()

pybind11_add_module(
  _cpp
  SHARED
  python_entry_point.cpp
  ${CPP_OBJECT_TARGETS}
)

message(${CPP_OBJECT_TARGETS})

target_compile_options(_cpp PRIVATE -g -Wall -Wextra)

target_precompile_headers(
    _cpp
    PRIVATE
    <pybind11/pybind11.h>
    <pybind11/numpy.h>
  )

add_subdirectory(tests)
