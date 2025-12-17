#pragma once

#include<stdint.h>

using u8  = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;

using i8  = int8_t;
using i16 = int16_t;
using i32 = int32_t;
using i64 = int64_t;

using f32 = float;
using f64 = double;

constexpr ssize_t SSIZE_ERROR = -1;
constexpr u32 FULL_MASK = 0xFFFFFFFF;

template<u8 DIMENSIONS>
struct Extent;

template<u8 DIMENSIONS>
struct Index;

template<u8 DIMENSIONS>
struct Point;

template<u8 DIMENSIONS>
struct Space;

template<u8 DIMENSIONS>
struct SquareMatrix;
