#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <cstdio>
#include <Eigen/Sparse>
#include <Eigen/Core>
#include <vector>
#include "hessone.h"
int add(int i, int j) {
    printf("Adding number C++ called from the python wrapper %d %d",i,j);
    return i + j;
}


PYBIND11_MODULE(Hess, m) {
    m.doc() = "Hessian in C++ wrapper for python"; // optional module docstring

    m.def("add", &add, "A function which adds two numbers");
    pybind11::class_<hessone>(m, "hessone")
        .def(pybind11::init<>())  // if the constructor takes a int double etc then it goes here <>
        .def("myfunc", &hessone::myfunc);
}
