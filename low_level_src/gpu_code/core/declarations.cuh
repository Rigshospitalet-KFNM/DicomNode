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

template<uint8_t DIMENSION>
struct Point;

template<uint8_t DIMENSIONS>
struct Index;

template<uint8_t DIMENSIONS>
struct Extent;

template<uint8_t DIMENSIONS>
struct SquareMatrix;

template<uint8_t DIMENSIONS>
class Space;
