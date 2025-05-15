
#include<stdint.h>

#include<array>
#include<utility>

#include"../declarations.hpp"

/**
 * @brief A point in a N-dimensional Space
 *
 * @tparam DIMENSIONS - The number of dimensions in the point.
 */
template<u8 DIMENSIONS>
struct Point {
  std::array<f32, DIMENSIONS> points{};

  Point() noexcept = default;


  template<typename T, size_t... idx_seq>
  Point(
    const T& arr, std::index_sequence<idx_seq...>
  ) noexcept : points{static_cast<f32>(arr[idx_seq])...} {}

  Point(Index<DIMENSIONS> idx)
    : Point(idx.coordinates, std::make_index_sequence<DIMENSIONS>{}) {}

  template<typename... Args>
  Point(Args... args) noexcept
    : points{static_cast<f32>(args)...} {
    static_assert(sizeof...(args) == DIMENSIONS);
  }

  template<typename T>
  f32& operator[](const T i){
    return points[i];
  }

  template<typename T>
  const f32& operator[](const T i) const {
    return points[i];
  }

  Point<DIMENSIONS> operator*(const SquareMatrix<DIMENSIONS>& m){
    Point<DIMENSIONS> v; // It's zero initialized!
    for(u8 j = 0; j < DIMENSIONS; j++){
      for(u8 i = 0; i < DIMENSIONS; i++){
        v[j] += points[i] * m[m.idx(j, i)];
      }
    }

    return v;
  }

  Point<DIMENSIONS> operator-(const Point<DIMENSIONS>& other) const {
    Point<DIMENSIONS> v; // It's zero initialized!
    for(u8 i = 0; i < DIMENSIONS; i++){
      v[i] = points[i] - other[i];
    }

    return v;
  }

  Point<DIMENSIONS> operator+(const Point<DIMENSIONS>& other) const {
    Point<DIMENSIONS> v; // It's zero initialized!
    for(u8 i = 0; i < DIMENSIONS; i++){
      v[i] = points[i] + other[i];
    }

    return v;
  }

  bool operator==(const Point<DIMENSIONS> other) const {
    bool ret = true;
    for(u8 i = 0; i < DIMENSIONS; i++){
      ret = ret && points[i] == other[i];
    }
    return ret;
  }

  static constexpr size_t elements() {
    return DIMENSIONS;
  }

  f32* begin() {
    return points.begin();
  }

  f32* end() {
    return points.end();
  }

  const f32* begin() const {
    return points.begin();
  }

  const f32* end() const {
    return points.end();
  }
};
