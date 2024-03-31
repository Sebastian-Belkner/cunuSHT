
cmake_minimum_required(VERSION 3.15...3.27)
project(popy LANGUAGES CXX CUDA) # Replace 'my_project' with the name of your project
find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)
# find_package(CUDAToolkit REQUIRED)
find_package(CUDA REQUIRED)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()


set(CMAKE_CUDA_STANDARD 14)

# add_library(po
#   pysht/c/pointing.cpp
#   pysht/c/pointing.cuh
#   pysht/c/pointing.cu
#   pysht/c/pointing.h
# )

# set_property(TARGET po PROPERTY CUDA_SEPARABLE_COMPILATION ON)
# target_compile_features(po PUBLIC cxx_std_11)


# Detect the installed nanobind package and import it into CMake
execute_process(COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NB_DIR)
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)


nanobind_add_module(popy
  pysht/c/pointing.cu
  pysht/c/pointing.h
  pysht/c/pointing.cpp
  pysht/c/pointing.cuh
)
# set_target_properties(popy PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# add_executable(popyexe pysht/c/pointing.cu)
 
# set_property(TARGET popyexe PROPERTY CUDA_SEPARABLE_COMPILATION ON)
# target_link_libraries(popyexe PRIVATE popy)

# Pass options to NVCC
# set(
#     CUDA_NVCC_FLAGS
# ${CUDA_NVCC_FLAGS};
#     -O3 -gencode --gpu-architecture sm_21 -std=c++11
#     )


# include_directories(
#     pysht/c
#     /mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include
#   )

# link_directories(
#   /mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/lib
#   pysht/c
#   )

# cuda_add_executables(pointingpy
#   pysht/c/pointing.cu
#   pysht/c/pointing.h
#   pysht/c/pointing.cpp
#   pysht/c/pointing.cuh
# )

# target_link_libraries(pointingpy -lcudart -lcuda)

# set(MY_PROJECT_SOURCES
#   pysht/c/pointing.cpp
#   ${MY_PROJECT_SOURCES}
#   pysht/c/pointing.cu
#   pysht/c/pointing.h
#   pysht/c/pointing.cuh
# )

# ADD_LIBRARY(MY_PROJECT_CUDA_KERNELS SHARED pysht/c/pointing.cu pysht/c/pointing.cpp) 
# SET_TARGET_PROPERTIES(MY_PROJECT_CUDA_KERNELS
#       PROPERTIES
#       VERSION "1"
#       LINKER_LANGUAGE "CUDA")
# ADD_EXECUTABLE(test pysht/c/pointing.cu)
# TARGET_LINK_LIBRARIES(test LINKER_LANGUAGE CXX MY_PROJECT_CUDA_KERNELS  LINKER_LANGUAGE CXX)

# set(MY_PROJECT_INCLUDES
# # Include folders are here + the cuda include directories
#   /mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/include
# )

# add_library(MY_PROJECT_CUDA_KERNELS SHARED pysht/c/pointing.cu)
# target_link_libraries(MY_PROJECT_CUDA_KERNELS PRIVATE ${MY_PROJECT_DEPENDENCIES} ${Python_LIBRARIES})
# target_include_directories(MY_PROJECT_CUDA_KERNELS PRIVATE ${MY_PROJECT_INCLUDES})
# set_target_properties(MY_PROJECT_CUDA_KERNELS
#   CMAKE_CUDA_CREATE_SHARED_LIBRARY ON
#   PROPERTIES
#   CUDA_STANDARD 17
#   CUDA_ARCHITECTURES 75
#   LINKER_LANGUAGE CUDA
#   POSITION_INDEPENDENT_CODE ON
#   CUDA_VISIBILITY_PRESET "hidden"
#   CUDA_SEPARABLE_COMPILATION ON
# )


# add_library(MY_PROJECT_LIBRARY SHARED ${MY_PROJECT_SOURCES})
# set_target_properties(MY_PROJECT_LIBRARY
#   PROPERTIES
#   CXX_STANDARD 17
#   POSITION_INDEPENDENT_CODE ON
#   CUDA_VISIBILITY_PRESET "hidden"
#   CUDA_SEPARABLE_COMPILATION ON
# )
# target_link_libraries(MY_PROJECT_LIBRARY PRIVATE ${MY_PROJECT_DEPENDENCIES} ${MY_PROJECT_CUDA_KERNELS})
# target_include_directories(MY_PROJECT_LIBRARY PRIVATE ${MY_PROJECT_INCLUDES})


# set_target_properties(pointingpy
#   PROPERTIES
#   CXX_STANDARD 17
#   POSITION_INDEPENDENT_CODE ON
#   CUDA_VISIBILITY_PRESET "hidden"
#   CUDA_SEPARABLE_COMPILATION ON
# )
# target_link_libraries(pointingpy PRIVATE ${MY_PROJECT_DEPENDENCIES} ${MY_PROJECT_LIBRARY})
# target_include_directories(pointingpy PRIVATE ${MY_PROJECT_INCLUDES})


# install(TARGETS pointingpy LIBRARY DESTINATION pointingpy)