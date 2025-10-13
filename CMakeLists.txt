cmake_minimum_required(VERSION 3.10)
project(Masterthesis LANGUAGES CXX)

# Make sure to set the CMAKE_PREFIX_PATH to where pybind11 is located (this path might be correct, but double-check)
set(CMAKE_PREFIX_PATH "C:\\Users\\T-Bone\\AppData\\Local\\Programs\\Python\\Python39\\lib\\site-packages\\pybind11\\share\\cmake\\pybind11")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find pybind11 package
find_package(pybind11 REQUIRED)

# Add the Python module using pybind11
pybind11_add_module(step_one prime_vars.cpp)
pybind11_add_module(step_two Step_Two.cpp)
pybind11_add_module(solver Solver.cpp)
pybind11_add_module(calc_flux Calc_flux.cpp)
pybind11_add_module(mesh_func Mesh_func.cpp)
pybind11_add_module(calc_courant Calc_courant.cpp)
pybind11_add_module(muscl_step MUSCL_one.cpp)



# Set the output name as ".pyd" (Python extension)
set_target_properties(step_one PROPERTIES SUFFIX ".pyd" LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR} LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR})
set_target_properties(step_two PROPERTIES SUFFIX ".pyd" LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR} LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR})
set_target_properties(solver PROPERTIES SUFFIX ".pyd" LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR} LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR})
set_target_properties(calc_flux PROPERTIES SUFFIX ".pyd" LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR} LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR})
set_target_properties(mesh_func PROPERTIES SUFFIX ".pyd" LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR} LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR})
set_target_properties(calc_courant PROPERTIES SUFFIX ".pyd" LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR} LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR})
set_target_properties(muscl_step PROPERTIES SUFFIX ".pyd" LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR} LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR})
