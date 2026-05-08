cmake_minimum_required(VERSION 3.10)
project(MatrixMultiplication_Lab2)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)

set(SOURCES
    src/main.cpp
    src/matrix_multiplier.cpp
)

add_executable(matrix_mult_lab2 ${SOURCES})

# Включение OpenMP для Visual Studio
if(MSVC)
    target_compile_options(matrix_mult_lab2 PRIVATE /O2 /EHsc /openmp)
    target_link_options(matrix_mult_lab2 PRIVATE /openmp)
else()
    find_package(OpenMP REQUIRED)
    target_compile_options(matrix_mult_lab2 PRIVATE -O3 -fopenmp)
    target_link_libraries(matrix_mult_lab2 OpenMP::OpenMP_CXX)
endif()