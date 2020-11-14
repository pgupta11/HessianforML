project(HessCalc)
cmake_minimum_required(VERSION 3.10)

add_subdirectory(thirdparty/pybind11)
include_directories(thirdparty/eigen)

pybind11_add_module(Hess hessWrapper.cpp hessone.h)