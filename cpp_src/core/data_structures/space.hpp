#pragma once

#include<stdint.h>

#include<pybind11/pybind11.h>

#include"../declarations.hpp"

template<u8 DIMENSIONS>
struct Space {
  Point<DIMENSIONS> starting_point;
  SquareMatrix<DIMENSIONS> basis;
  SquareMatrix<DIMENSIONS> inverted_basis;
  Extent<DIMENSIONS> extent;

  static constexpr const char* starting_point_attr_name = "starting_point";
  static constexpr const char* basis_attr_name = "basis";
  static constexpr const char* inverted_basis_attr_name = "inverted_basis";
  static constexpr const char* extent_attr_name = "extent";

  Point<DIMENSIONS> at_index(const Index<DIMENSIONS>& index) const {
    Point<DIMENSIONS> point{index};
    return point * basis + starting_point;
  }

  Point<DIMENSIONS> interpolate_point(const Point<DIMENSIONS>& point) const {
    return (point - starting_point) * inverted_basis;
  }

  constexpr size_t elements() const {
    return extent.elements();
  }
};
