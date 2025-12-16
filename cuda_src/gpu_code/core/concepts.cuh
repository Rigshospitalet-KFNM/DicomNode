#pragma once
#include<concepts>
#include<stdint.h>
#include<type_traits>

#include"declarations.cuh"

template<typename OP, typename T>
concept CBinaryOperator = requires (OP op, T a, T b) {
  { OP::apply(a, b) } -> std::same_as<T>;
  { OP::equals(a, b) } -> std::same_as<bool>;
  { OP::identity() } -> std::same_as<T>;
  { OP::remove_volatile(a)} -> std::same_as<T>;
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

template<typename Container>
concept dicomnodeContainer = requires(Container t) {
  { t.elements() } noexcept -> std::same_as<size_t>;
};

// This concept doesn't really make that much sense since, there's only one
// extent class, and I am unsure that could be another
template<template<uint8_t DIMENSION> typename EXTENT, uint8_t DIMENSION>
concept CExtent = requires(EXTENT<DIMENSION> extent, Index<DIMENSION> idx) {
  {extent.contains(idx)} -> std::same_as<bool>;
};

// I swear the syntax
template<template<uint8_t DIMENSIONS, typename T> typename VOLUME, uint8_t DIMENSIONS, typename T>
concept CVolume = requires(VOLUME<DIMENSIONS, T> volume, Index<DIMENSIONS> Index) {
  { volume(Index)} -> std::same_as<T>;
  { volume.elements() } -> std::same_as<size_t>;
  { volume.extent() } -> std::same_as<const Extent<DIMENSIONS>&>;
};

template<typename T, uint8_t DIMENSIONS>
concept CImage = requires(T image, Point<DIMENSIONS> p){
  //{ image(p) } -> std::same_as<T>;
  { image.elements() } -> std::same_as<size_t>;
  { image.extent() } -> std::same_as<const Extent<DIMENSIONS>&>;
};
