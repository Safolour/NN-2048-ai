cmake_minimum_required(VERSION 3.20)
project(m3_tuple_backend LANGUAGES CXX)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_m3_tuple_backend m3_tuple_backend.cpp)
set_target_properties(_m3_tuple_backend PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)

if(MSVC)
    target_compile_options(_m3_tuple_backend PRIVATE /O2 /EHsc)
else()
    target_compile_options(_m3_tuple_backend PRIVATE -O3)
endif()
