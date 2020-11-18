project(HessCalc)
cmake_minimum_required(VERSION 3.10)

add_subdirectory(thirdparty/pybind11)
include_directories(thirdparty/eigen)
include_directories(thirdparty/xtensor/include)
include_directories(thirdparty/xtensor-python/include)
include_directories(thirdparty/xtl/include)
pybind11_add_module(Hess hessWrapper.cpp hessone.h)