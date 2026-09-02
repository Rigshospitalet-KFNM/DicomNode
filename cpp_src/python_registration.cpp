//
// Created by christoffer on 9/2/26.
//

#include "core/core.hpp"
#include "python_registration.hpp"

namespace {
  constexpr f32 abs(const f32& x) {
    return x > 0 ? x : -x;
  }


}


/** Compares the difference
 *
 * @param a
 * @param b
 * @return
 */
constexpr f32 compare_volumes(const Volume<f32, 3>& a, const Volume<f32, 3>& b) {
  f32 error = 0.0f;

  for (u64 i = 0; i < a.elements(); i++) {
    error += abs(a.at(i) - b.at(i));
  }

  return error;
}