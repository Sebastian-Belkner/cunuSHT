cmake_minimum_required(VERSION 3.15...3.27)
project(popy LANGUAGES CXX CUDA)
find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(CUDA REQUIRED)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()



# set(CMAKE_CUDA_STANDARD 14)


execute_process(COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NB_DIR)
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)

enable_language(CUDA)
nanobind_add_module(popy
  cunusht/c/pointing.cu
  cunusht/c/pointing.h
  cunusht/c/pointing.cpp
  cunusht/c/pointing.cuh
)

nanobind_add_module(dopy
  cunusht/c/doubling.cu
  cunusht/c/doubling.h
  cunusht/c/doubling.cpp
  cunusht/c/doubling.cuh
)

# Check if the compiler supports C++17
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++17" COMPILER_SUPPORTS_CXX17)

if(COMPILER_SUPPORTS_CXX17)
    # Use C++17 if the compiler supports it
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    target_compile_features(popy PUBLIC cxx_std_17)
    message(STATUS "C++17 support enabled.")
else()
    # Use an older C++ standard if C++17 is not supported
    message(WARNING "The compiler does not support C++17. Using an older standard.")
endif()


install(TARGETS popy LIBRARY DESTINATION interface)