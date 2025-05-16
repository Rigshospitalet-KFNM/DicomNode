
enable_testing()

include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG 52eb810
)

FetchContent_MakeAvailable(googletest)

add_executable(cpp_tests tests_initiator.cpp)
target_link_libraries(
  cpp_tests
  GTest::gtest_main
  pybind11::module
  Python::Python
)
include(GoogleTest)
gtest_discover_tests(cpp_tests)
