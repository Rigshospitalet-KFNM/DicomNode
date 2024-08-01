#pragma once
#include<concepts>
#include<stdint.h>

template<typename OP, typename T>
concept CBinaryOperator = requires (OP op, T a, T b) {
  {OP::apply(a, b) } -> std::same_as<T>;
  {OP::equals(a, b) } -> std::same_as<bool>;
  {OP::identity() } -> std::same_as<T>;
  {OP::remove_volatile(a)} -> std::same_as<T>;
};

template<typename OP, typename T_IN, typename T_OUT, typename... Args>
concept Mapping = requires (OP op, T_IN a, uint64_t flat_index, Args... args){
  {OP::map_to(a, flat_index, args...)} -> std::same_as<T_OUT>;
};

template<typename OP, typename T, typename... Args>
concept Mirrors = requires (OP op, const T* a, uint64_t idx, Args... args){
  {OP::mirrors(a, idx, args...)} -> std::same_as<T>;
};

template<typename OP, typename T_IN, typename T_OUT, typename... Args>
concept MappingBinaryOperator = (
  CBinaryOperator<OP, T_OUT> &&
  Mapping<OP, T_IN, T_OUT, Args...>
);
