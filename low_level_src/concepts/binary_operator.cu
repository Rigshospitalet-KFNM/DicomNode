#ifndef DICOMNODE_BINARY_OPERATOR
#define DICOMNODE_BINARY_OPERATOR

#include<concepts>

template<typename op, typename T>
concept CBinaryOperator = requires (op OP, T a, T b) {
  {op::apply(a, b) } -> std::same_as<T>;
  {op::equals(a, b) } -> std::same_as<bool>;
  {op::identity() } -> std::same_as<T>;
  {op::remove_volatile(a)} -> std::same_as<T>;
};

template<typename op, typename T_IN, typename T_OUT, typename... Args>
concept MappingBinaryOperator = requires (op op, T_OUT a, T_OUT b, T_IN c, Args... args) {
  {op::apply(a, b) } -> std::same_as<T_OUT>;
  {op::equals(a, b) } -> std::same_as<bool>;
  {op::identity() } -> std::same_as<T_OUT>;
  {op::remove_volatile(a)} -> std::same_as<T_OUT>;
  {op::map_to(c, args...)} -> std::same_as<T_OUT>;
};



#endif
