#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>

void interpolate_linear(const pybind11::object& image, const pybind11::object new_space){
  const pybind11::object& original_space = image.attr("space");


}
