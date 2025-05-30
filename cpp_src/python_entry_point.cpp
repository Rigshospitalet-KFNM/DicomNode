/**
 * @file python_entry_point.cpp
 * @author your name (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2025-05-14
 *
 * @copyright Copyright (c) 2025
 *
 */

#include<string>

#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>

#include"core/core.hpp"

#include"python_interpolation.hpp"

PYBIND11_MODULE(_cpp, m){
  m.doc() = "Low level module with CPU code";

  pybind11::class_<CppError_t>(m, "CppError")
    .def("__bool__", [](const CppError_t& error){
      return error != SUCCESS;
    })
    .def("__str__", [](const CppError_t& error){
      return std::to_string(error);
    });

  apply_interpolation_module(m);
}
