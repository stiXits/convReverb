cmake_minimum_required(VERSION 3.9)
project(impl)

set(CMAKE_CXX_STANDARD 11)

add_executable(impl
        CPUconv.cpp
        CPUconv.hpp
        CPUconvIdentity.cpp
        CPUconvIdentity.h
        CPUconvOAReverb.cpp
        CPUconvOAReverb.h
        GPUconvOAReverb.cpp
        GPUconvOAReverb.h
        CPUconvSimpleReverb.cpp
        CPUconvSimpleReverb.h
        main.cpp)

target_link_libraries(impl sndfile fftw3 clFFT arm_compute arm_compute_core OpenCL)
