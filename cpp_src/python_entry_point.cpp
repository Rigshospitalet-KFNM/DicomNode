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

#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>

PYBIND11_MODULE(_cpp, m){
  m.doc() = "Low level module with CPU code";
}
