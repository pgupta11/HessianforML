#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <cstdio>
#include <Eigen/Sparse>
#include <Eigen/Core>
#include <vector>
#include "hessone.h"

PYBIND11_MODULE(Hess, m) {
    m.doc() = "Hessian in C++ wrapper for python"; // optional module docstring

    //m.def("add", &add, "A function which adds two numbers");
    pybind11::class_<hessone>(m, "hessone")
        .def(pybind11::init<int>())  // if the constructor takes a int double etc then it goes here <>
        .def("myfunc", &hessone::myfunc)
        .def("add", &hessone::add);
}
