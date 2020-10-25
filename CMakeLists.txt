# ----------------------------------------------------------------------------
#  Root CMake file for DfD Graph Cuts
# ----------------------------------------------------------------------------
#
# To perform an out of source build do the following on the command line:
# 1. navigate to the root project directory (in this case DfD2)
# 2. mkdir build
# 3. cd build
# 4. depending on your system (windows/linux) type the following
# Example CMake build line for windows & VS2015
# windows: cmake -G "Visual Studio 14 2015 Win64" -DUSE_AVX_INSTRUCTIONS=ON ..
# linux: cmake -DUSE_AVX_INSTRUCTIONS=ON ..
#
# 5. To compile:
# windows: cmake --build . --config Release
# linux: cmake --build . --config Release -- -j4
#


cmake_minimum_required(VERSION 2.8.12)

#set(CMAKE_CXX_STANDARD 11)
#set(CMAKE_CXX_STANDARD_REQUIRED ON)

PROJECT(PlayGround_Project)
set(exe_name "pg")

# This looks for an environment variable called PLATFORM.  This can be used to point
# cmake to various locations based on the individual machines that the code is complied on
set(PLATFORM "$ENV{PLATFORM}" CACHE INTERNAL "Copied from environment variable")
message("Platform: " ${PLATFORM})

#if(WIN32)
#	include(D:/dlib/dlib/cmake)
#else()
#    if(${PLATFORM} STREQUAL "HPC")
#        include(~/dlib/dlib/cmake)
#    elseif(${PLATFORM} STREQUAL "SL02319")
#        include(/mnt/data/dlib-19.17/dlib/cmake)
#    else()
#        include(~/dlib/dlib/cmake)
#    endif()
#endif()
if(WIN32)
    if(${PLATFORM} STREQUAL "Laptop")
        set(COMMON_ROOT "D:/Common")
        set(DLIB_CONTRIB "D:/Projects/dlib-contrib")
        set(DFD_COMMON_ROOT "D:/Projects/dfd_common")
        include(D:/dlib-19.17/dlib/cmake)
    elseif(${PLATFORM} STREQUAL "LaptopN")
        set(COMMON_ROOT "C:/Projects/Common")
        set(DLIB_CONTRIB "C:/Projects/dlib-contrib")
        set(DFD_COMMON_ROOT "C:/Projects/dfd_common")
        include(C:/dlib-19.17/dlib/cmake)
    elseif(${PLATFORM} STREQUAL "BEAST")
        set(COMMON_ROOT "D:/Common")
        set(DLIB_CONTRIB "D:/Projects/dlib-contrib")
        set(DFD_COMMON_ROOT "D:/Projects/dfd_common")
        include(D:/dlib-19.19/dlib/cmake)    
    else()
        set(COMMON_ROOT "D:/Common")
        set(DLIB_CONTRIB "D:/Projects/dlib-contrib")
        set(DFD_COMMON_ROOT "D:/Projects/dfd_common")
        include(D:/dlib-19.19/dlib/cmake)
    endif()
    
else()
    if(${PLATFORM} STREQUAL "MainGear")
        set(_ECLIPSE_VERSION "3.8")
        set(COMMON_ROOT "~/Common")
        set(DLIB_CONTRIB "~/Projects/dlib-contrib")
        set(DFD_COMMON_ROOT "~/Projects/dfd_common")
        include(~/dlib-19.17/dlib/cmake)
    elseif(${PLATFORM} STREQUAL "HPC")
        set(COMMON_ROOT "$HOME/Common")
        set(DLIB_CONTRIB "~/Projects/dlib-contrib")
        set(DFD_COMMON_ROOT "$HOME/Projects/dfd_common")
        include(~/dlib-19.19/dlib/cmake) 
    else()
        set(COMMON_ROOT "~/Projects/Common")
        set(DLIB_CONTRIB "~/Projects/dlib-contrib")
        set(DFD_COMMON_ROOT "~/Projects/dfd")
        include(~/dlib-19.19/dlib/cmake)
    endif()  
    	
endif()


include(${COMMON_ROOT}/cmake_utils/use_cpp_11.cmake)
include(${COMMON_ROOT}/cmake_utils/use_sse_avx.cmake)

# look for any *.c and *.cpp files
file(GLOB SOURCES "src/*.c*"
    "${COMMON_ROOT}/src/*.cpp"
    "${DLIB_CONTRIB}/src/*.cpp"
    )

set(project_includes "include" 
    "${COMMON_ROOT}/include" 
    "${DLIB_CONTRIB}/include" 
    "${DFD_COMMON_ROOT}/include")

message(" ")
message("--------------------------------------------------------")
message("Source Files: ")
foreach(S ${SOURCES})
    message("  " ${S})
endforeach()
message("--------------------------------------------------------")
message(" ")

# find OpenCV
find_package(OpenCV REQUIRED)
if (OpenCV_FOUND)
	set(project_includes ${project_includes} ${OpenCV_INCLUDE_DIRS})
else()
	message("OpenCV not found, so we won't build the project.")
endif()

message(" ")
include(${COMMON_ROOT}/cmake_utils/find_edtpdv.cmake)
if(EDTPDV_FOUND)
    set(project_includes ${project_includes} ${EDTPDV_INCLUDE_DIRS})
    set(project_libs ${project_libs} ${EDTPDV_LIBS})
endif()

#add_definitions(-DMATPLOTLIBCPP_PYTHON_HEADER=Python.h)
#set(project_includes ${project_includes} D:/Python_3.6.5/include)
#set(project_includes ${project_includes} D:/Python_3.6.5/Lib/site-packages/numpy/core/include)

message(" ")
message("--------------------------------------------------------")
message("Project Includes: ")
foreach(S ${project_includes})
    message("  " ${S})
endforeach()
message("--------------------------------------------------------")
message(" ")

# link_directories(D:/Python_3.6.5/libs)

include_directories(${project_includes})
add_executable(${exe_name} ${SOURCES})
target_link_libraries(${exe_name} dlib::dlib ${OpenCV_LIBS})

if(WIN32)
    message("MSVC Version: " ${MSVC_VERSION})
    # uncomment for MSVC and cmake greater than 3.6.0
    set(default_build_type "Release")
    if(MSVC AND (${CMAKE_VERSION} VERSION_GREATER "3.6.0"))
        target_compile_options(${exe_name} PRIVATE "/MP")
        set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${exe_name})
    endif()

endif()

message(" ")
message(STATUS "Build Complete!")
message(" ")



