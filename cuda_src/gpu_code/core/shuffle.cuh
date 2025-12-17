/**
 * @file shuffle.cuh
 * @author Christoffer (cjen0668@regionh.dk)
 * @brief This is an experiment to use the shuffle to move around larger
 * datastructers,
 * @version 0.1
 * @date 2025-12-16
 *
 * @copyright Copyright (c) 2025
 *
 */

#pragma once

#include<stdint.h>

#include"declarations.hpp"



template<typename T>
T shuffle_up(T* val, i32 delta){
  static_assert(false, "You need to specialize this function");
}


/* THIS DOESN'T WORK BECAUSE OF MISALIGNED ADDRESS */

#ifdef FALSE

/**
 * @brief Function for shuffling any data structure for not limited by
 * fundamental types which the __shfl_up operator is limited to
 *
 * Also it's slows as fuck!
 *
 * @tparam T - Type with
 * @param share
 * @param delta
 * @return __device__
 */
template<typename T>
__device__ T shfl_up(T* share, uint32_t delta){
  static_assert(sizeof(T) > 0);
  static_assert(sizeof(T) < UINT16_MAX);
  T ret;

  int16_t bytes_shifted = 0;
  while (bytes_shifted < sizeof(T)){
    if(sizeof(T) - bytes_shifted > sizeof(uint64_t)){
      using ShiftType = uint64_t;
      int16_t shift = bytes_shifted / sizeof(ShiftType);
      ShiftType *source = (ShiftType*)share + shift;
      ShiftType *destination = (ShiftType*)&ret + shift;
      *destination = __shfl_up_sync(FULL_MASK, *source, delta);
      bytes_shifted += sizeof(ShiftType);
    } else if(sizeof(T) - bytes_shifted > sizeof(uint32_t)) {
      using ShiftType = uint32_t;
      int16_t shift = bytes_shifted / sizeof(ShiftType);
      ShiftType *source = (ShiftType*)share + shift;
      ShiftType *destination = (ShiftType*)&ret + shift;
      *destination = __shfl_up_sync(FULL_MASK, *source, delta);
      bytes_shifted += sizeof(ShiftType);
    } else if(sizeof(T) - bytes_shifted > sizeof(uint16_t)) {
      using ShiftType = uint16_t;
      int16_t shift = bytes_shifted / sizeof(ShiftType);
      ShiftType *source = (ShiftType*)share + shift;
      ShiftType *destination = (ShiftType*)&ret + shift;
      *destination = __shfl_up_sync(FULL_MASK, *source, delta);
      bytes_shifted += sizeof(ShiftType);
    } else {
      using ShiftType = uint8_t;
      int16_t shift = bytes_shifted;
      ShiftType *source = (ShiftType*)share + shift;
      ShiftType *destination = (ShiftType*)&ret + shift;
      *destination = __shfl_up_sync(FULL_MASK, *source, delta);
      bytes_shifted += sizeof(ShiftType);
    }
  }

  return ret;
}


// There's UB here, but it's FUCKING FAAAAAAST

template<typename T, int BytesShifted = 0>
__device__ __forceinline__ void _shuffle_bytes(T* share, T* ret, int delta) {
  //
  // static_assert(std::is_trivially_copyable_v<T>); // This is technical true, but I need some operators...
  //static_assert(alignof(T) >= 8); I'll try and write some test that fucks this up.

  if constexpr (BytesShifted >= sizeof(T)) {
    return; // Base case
  } else if constexpr (sizeof(T) - BytesShifted >= sizeof(uint64_t)) {
    using ShiftType = uint64_t;
    constexpr int shift = BytesShifted / sizeof(ShiftType);
    ShiftType *source = (ShiftType*)share + shift;
    ShiftType *destination = (ShiftType*)ret + shift;
    *destination = __shfl_up_sync(FULL_MASK, *source, delta);
    _shuffle_bytes<T, BytesShifted + sizeof(uint64_t)>(share, ret, delta);
  } else if constexpr (sizeof(T) - BytesShifted >= sizeof(uint32_t)) {
    using ShiftType = uint32_t;
    constexpr int shift = BytesShifted / sizeof(ShiftType);
    ShiftType *source = (ShiftType*)share + shift;
    ShiftType *destination = (ShiftType*)ret + shift;
    *destination = __shfl_up_sync(FULL_MASK, *source, delta);
    _shuffle_bytes<T, BytesShifted + sizeof(uint32_t)>(share, ret, delta);
  } else if constexpr (sizeof(T) - BytesShifted >= sizeof(uint16_t)) {
    using ShiftType = uint16_t;
    constexpr int shift = BytesShifted / sizeof(ShiftType);
    ShiftType *source = (ShiftType*)share + shift;
    ShiftType *destination = (ShiftType*)ret + shift;
    *destination = __shfl_up_sync(FULL_MASK, *source, delta);
    _shuffle_bytes<T, BytesShifted + sizeof(uint16_t)>(share, ret, delta);
  } else {
    using ShiftType = uint8_t;
    constexpr int shift = BytesShifted;
    ShiftType *source = (ShiftType*)share + shift;
    ShiftType *destination = (ShiftType*)ret + shift;
    *destination = __shfl_up_sync(FULL_MASK, *source, delta);
    _shuffle_bytes<T, BytesShifted + sizeof(uint8_t)>(share, ret, delta);
  }
}

template<typename T>
__device__ __forceinline__ T shuffle_bytes(T* share, int delta){
  T ret;
  _shuffle_bytes<T>(share, &ret, delta);
  return ret;
}

/**
 * @brief
 *
 * template<typename T, int BytesShifted = 0>
__device__ __forceinline__ void shuffle_bytes_3(T* share, T* ret, int delta) {
  if constexpr (BytesShifted >= sizeof(T)) {
    return; // Base case
  } else if constexpr (sizeof(T) - BytesShifted >= sizeof(uint64_t)) {
    using ShiftType = uint64_t;
    constexpr int shift = BytesShifted / sizeof(ShiftType);

    // Read via bit_cast from byte array
    ShiftType source_val = cuda::std::bit_cast<ShiftType>(
      *reinterpret_cast<cuda::std::array<unsigned char, sizeof(ShiftType)>*>(
        reinterpret_cast<unsigned char*>(share) + BytesShifted
      )
    );

    ShiftType shuffled = __shfl_up_sync(FULL_MASK, source_val, delta);

    // Write via bit_cast to byte array
    auto bytes = cuda::std::bit_cast<cuda::std::array<unsigned char, sizeof(ShiftType)>>(shuffled);
    *reinterpret_cast<cuda::std::array<unsigned char, sizeof(ShiftType)>*>(
      reinterpret_cast<unsigned char*>(ret) + BytesShifted
    ) = bytes;

    shuffle_bytes_3<T, BytesShifted + sizeof(uint64_t)>(share, ret, delta);

  } else if constexpr (sizeof(T) - BytesShifted >= sizeof(uint32_t)) {
    using ShiftType = uint32_t;

    ShiftType source_val = cuda::std::bit_cast<ShiftType>(
      *reinterpret_cast<cuda::std::array<unsigned char, sizeof(ShiftType)>*>(
        reinterpret_cast<unsigned char*>(share) + BytesShifted
      )
    );

    ShiftType shuffled = __shfl_up_sync(FULL_MASK, source_val, delta);

    auto bytes = cuda::std::bit_cast<cuda::std::array<unsigned char, sizeof(ShiftType)>>(shuffled);
    *reinterpret_cast<cuda::std::array<unsigned char, sizeof(ShiftType)>*>(
      reinterpret_cast<unsigned char*>(ret) + BytesShifted
    ) = bytes;

    shuffle_bytes_3<T, BytesShifted + sizeof(uint32_t)>(share, ret, delta);

  } else if constexpr (sizeof(T) - BytesShifted >= sizeof(uint16_t)) {
    using ShiftType = uint16_t;

    ShiftType source_val = cuda::std::bit_cast<ShiftType>(
      *reinterpret_cast<cuda::std::array<unsigned char, sizeof(ShiftType)>*>(
        reinterpret_cast<unsigned char*>(share) + BytesShifted
      )
    );

    ShiftType shuffled = __shfl_up_sync(FULL_MASK, source_val, delta);

    auto bytes = cuda::std::bit_cast<cuda::std::array<unsigned char, sizeof(ShiftType)>>(shuffled);
    *reinterpret_cast<cuda::std::array<unsigned char, sizeof(ShiftType)>*>(
      reinterpret_cast<unsigned char*>(ret) + BytesShifted
    ) = bytes;

    shuffle_bytes_3<T, BytesShifted + sizeof(uint16_t)>(share, ret, delta);

  } else {
    using ShiftType = uint8_t;

    ShiftType source_val = *reinterpret_cast<unsigned char*>(
      reinterpret_cast<unsigned char*>(share) + BytesShifted
    );

    ShiftType shuffled = __shfl_up_sync(FULL_MASK, source_val, delta);

    *reinterpret_cast<unsigned char*>(
      reinterpret_cast<unsigned char*>(ret) + BytesShifted
    ) = shuffled;

    shuffle_bytes_3<T, BytesShifted + sizeof(uint8_t)>(share, ret, delta);
  }
}
 *
 */

#endif