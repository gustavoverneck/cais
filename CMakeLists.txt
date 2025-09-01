cmake_minimum_required(VERSION 3.10)
project(cais)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include directories
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/third_party/eigen-3.4.0
    ${PROJECT_SOURCE_DIR}/third_party/opencl_wrapper
    ${PROJECT_SOURCE_DIR}/third_party/opencl_wrapper/OpenCL/include
)

# Source files
set(SOURCES
    src/cpu_engine.cpp
    src/kernels.cpp
    src/opencl_engine.cpp
    src/solver.cpp
)

add_executable(cais examples/main.cpp ${SOURCES})


# Find and link OpenCL
find_package(OpenCL REQUIRED)
if(OpenCL_FOUND)
    target_include_directories(cais PRIVATE ${OpenCL_INCLUDE_DIRS})
    target_link_libraries(cais PRIVATE ${OpenCL_LIBRARIES})
else()
    message(FATAL_ERROR "OpenCL not found")
endif()

# Find all example source files
file(GLOB EXAMPLES "examples/*.cpp")

foreach(example_file ${EXAMPLES})
    get_filename_component(example_name ${example_file} NAME_WE)
    add_executable(${example_name} ${example_file} ${SOURCES})
    target_include_directories(${example_name} PRIVATE ${OpenCL_INCLUDE_DIRS})
    target_link_libraries(${example_name} PRIVATE ${OpenCL_LIBRARIES})
endforeach()