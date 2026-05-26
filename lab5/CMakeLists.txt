cmake_minimum_required(VERSION 3.15)
project(Laba_3_MPI LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(MPI_CXX_SKIP_MPICXX TRUE)
find_package(MPI REQUIRED)

add_executable(matrix_mul matrix_mul.cpp)
target_link_libraries(matrix_mul PRIVATE MPI::MPI_CXX)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
  target_compile_options(matrix_mul PRIVATE -Wall -Wextra -Wpedantic -O2)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  target_compile_options(matrix_mul PRIVATE /W4 /utf-8 /O2)
endif()