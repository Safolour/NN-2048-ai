cmake_minimum_required(VERSION 3.20)
project(m2_fast_backend LANGUAGES CXX)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(
    _m2_fast_backend
    m2_fast_backend.cpp
    movement_core.cpp
)

target_include_directories(_m2_fast_backend PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(
    _m2_fast_backend
    PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)

if(MSVC)
    target_compile_options(_m2_fast_backend PRIVATE /O2 /EHsc)
else()
    target_compile_options(_m2_fast_backend PRIVATE -O3)
endif()
