#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>

inline constexpr int ARRAY_FLAGS = pybind11::array_t<int>::forcecast
                                 | pybind11::array_t<int>::c_style;