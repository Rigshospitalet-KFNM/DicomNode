#pragma once

#include<stdint.h>

#include<array>

#include"../declarations.hpp"

template<u8 DIMENSIONS>
struct SquareMatrix {
  using SQUARE_MATRIX_TYPE = f32;

  std::array<f32, DIMENSIONS * DIMENSIONS> points;

  static constexpr u32 idx(const i32 row, const i32 col){
    return row * DIMENSIONS + col;
  }

  f32& operator[](const i32 i){
    return points[i];
  }

  f32& operator[](const u32 i){
    return points[i];
  }

  const f32& operator[](const i32 i) const {
    return points[i];
  }

  const f32& operator[](const u32 i) const {
    return points[i];
  }

  f32* begin(){
    return points.begin();
  }

  f32* end(){
    return points.end();
  }

  const Point<DIMENSIONS> operator*(
    const Point<DIMENSIONS>& other
  ) const {
      // It's zero initialized!
      Point<DIMENSIONS> point;
      for(u8 j = 0; j < DIMENSIONS; j++){
        for(u8 i = 0; i < DIMENSIONS; i++){
          point[j] += other[i] * points[idx(i,j)];
        }
      }

    return point;
  }

  static constexpr size_t elements() {
    return DIMENSIONS * DIMENSIONS;
  }
};