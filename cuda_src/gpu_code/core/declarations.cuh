/**
 * @file declarations.cuh
 * @author Demiguard
 * @brief This header file is to the declarations of all data structures used
 * throughout the library such that they can be referenced in implementation
 * @version 0.1
 * @date 2025-02-27
 *
 * @copyright Copyright (c) 2025 Rigshospitalet - MIT
 *
 */
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

template<u8 DIMENSION>
struct Point;

template<u8 DIMENSIONS>
struct Index;

template<u8 DIMENSIONS>
struct Extent;

template<u8 DIMENSIONS>
struct SquareMatrix;

template<u8 DIMENSIONS>
class Space;
