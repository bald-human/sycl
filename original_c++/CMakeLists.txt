cmake_minimum_required(VERSION 3.23.5)
project(Masterthesis LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set pybind11 path FIRST
set(CMAKE_PREFIX_PATH "/groups/astro/tomdonko/.local/lib/python3.8/site-packages/pybind11/share/cmake/pybind11")

# Find pybind11
find_package(pybind11 REQUIRED)

# Create the Python module using pybind11
pybind11_add_module(muscl_step MUSCL_one.cpp)

# MANUAL SYCL setup - bypass IntelSYCL package issues
target_compile_options(muscl_step PRIVATE -fsycl)
target_link_options(muscl_step PRIVATE -fsycl)

# Set output directory (optional)
set_target_properties(muscl_step PROPERTIES 
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}
    LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR}
)