#pragma once

#include<stdint.h>

constexpr uint32_t FULL_MASK = 0xFFFFFFFF;

template<typename T>
__device__ T shfl_up(T* share, uint32_t delta){
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
    } else  {
      using ShiftType = uint8_t;
      int16_t shift = bytes_shifted / sizeof(ShiftType);
      ShiftType *source = (ShiftType*)share + shift;
      ShiftType *destination = (ShiftType*)&ret + shift;
      *destination = __shfl_up_sync(FULL_MASK, *source, delta);
      bytes_shifted += sizeof(ShiftType);
    }
  }

  return ret;
}