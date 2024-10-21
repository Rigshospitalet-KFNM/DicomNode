#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>

#include<iostream>

using basis_t = pybind11::array_t<float>;
using domain_array = pybind11::array_t<int>;



void interpolate_linear_templated(){

}

void interpolate_linear(const pybind11::object& image,
                        const pybind11::object new_space,
                        const domain_array& new_domain
  ){
  const pybind11::object& original_space = image.attr("space");
  const basis_t& inverted_basis = pybind11::cast<basis_t>(original_space.attr("inverted_raw"));
  const pybind11::buffer_info& inverted_basis_buffer = inverted_basis.request(false);

  if(inverted_basis_buffer.ndim != 2){
    throw std::runtime_error("The Basis is not a 3 by 3 matrix");
  }


}

void apply_interpolation_module(pybind11::module& m){
  pybind11::module sub_module = m.def_submodule(
    "interpolation",
    "This module contains functions for resampling and interpolation.\n"
  );

  sub_module.def("linear_interpolate", &interpolate_linear);
}