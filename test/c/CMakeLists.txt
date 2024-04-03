
cmake_minimum_required(VERSION 3.15...3.27)
project(popy LANGUAGES CXX CUDA)
find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(CUDA REQUIRED)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()


set(CMAKE_CUDA_STANDARD 14)


execute_process(COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NB_DIR)
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)


nanobind_add_module(popy
  test_pointing.cu
  test_pointing.h
  test_pointing.cpp
  test_pointing.cuh
)