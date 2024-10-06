cmake_minimum_required(VERSION 3.10)

# Set the project name
project(SpectroMat)

set (CMAKE_CXX_STANDARD 17)

# Find the OpenCV package
find_package(OpenCV REQUIRED)

# Include the OpenCV headers
include_directories(${OpenCV_INCLUDE_DIRS})

# Add the executable
add_executable(SpectroMat main.cpp "spectromat.h" "spectromat.cpp")

# Link the OpenCV libraries to the executable
target_link_libraries(SpectroMat ${OpenCV_LIBS})
