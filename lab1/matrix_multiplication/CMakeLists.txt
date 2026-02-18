cmake_minimum_required(VERSION 3.10)
project(MatrixMultiplication)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Опции компиляции для оптимизации
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -funroll-loops")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

# Директории с заголовочными файлами
include_directories(include)

# Исходные файлы
set(SOURCES
    src/main.cpp
    src/matrix_multiplier.cpp
)

# Исполняемый файл
add_executable(matrix_mult ${SOURCES})

# Для параллельной версии (OpenMP)
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(matrix_mult OpenMP::OpenMP_CXX)
    target_compile_options(matrix_mult PRIVATE ${OpenMP_CXX_FLAGS})
    message(STATUS "OpenMP found - parallel version enabled")
    
    # Дополнительные оптимизации для OpenMP
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_compile_options(matrix_mult PRIVATE -fopenmp-simd)
    endif()
else()
    message(STATUS "OpenMP not found - using sequential version")
endif()

# Векторизация
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(matrix_mult PRIVATE -ftree-vectorize -ffast-math)
endif()