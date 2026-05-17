set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --time compile_times.csv")

include(FetchContent)

if(NOT EXISTS "${CMAKE_BINARY_DIR}/cmake/CPM.cmake")
    file(
        DOWNLOAD
        https://github.com/cpm-cmake/CPM.cmake/releases/download/v0.42.0/CPM.cmake
        "${CMAKE_BINARY_DIR}/cmake/CPM.cmake"
    )
endif()

include("${CMAKE_BINARY_DIR}/cmake/CPM.cmake")

CPMAddPackage(
        NAME googletest
        GITHUB_REPOSITORY google/googletest
        VERSION 1.14.0
        OPTIONS
        "INSTALL_GTEST OFF"
        "gtest_force_shared_crt ON"
)

enable_testing()
include(GoogleTest)

file(GLOB_RECURSE CUDA_TEST_SOURCES "tests/*.cu")

foreach(test_src ${CUDA_TEST_SOURCES})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} ${test_src})
    target_link_libraries(${test_name} PRIVATE
            GTest::gtest_main
            pybind11::module
            Python::Python
            utilities)
    gtest_discover_tests(${test_name})
endforeach()

#  # Sadly Coverage for GPU code is not supported...
#  # https://docs.nvidia.com/cuda/cuda-features-archive/index.html#bullseye-support
