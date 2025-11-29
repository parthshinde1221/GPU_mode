cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(GPU_mode LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Default build type
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# All executables go under build/bin/
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# CUDA architectures (tweak later if needed)
set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86)

# Make include/ visible to all targets
include_directories(${CMAKE_SOURCE_DIR}/include)

# Add your first kernel project
# add_subdirectory(matrix_multiplication)
# add_subdirectory(vector_add)
add_subdirectory(reduction)
