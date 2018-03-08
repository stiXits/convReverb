cmake_minimum_required(VERSION 3.9)
project(impl)

set(CMAKE_CXX_STANDARD 11)
find_package(Boost COMPONENTS system filesystem REQUIRED)


add_executable(impl
        CPUconvIdentity.cpp
        CPUconvIdentity.h
        CPUconvOAReverb.cpp
        CPUconvOAReverb.h
        GPUconvOAReverb.cpp
        GPUconvOAReverb.h
        CPUconvSimpleReverb.cpp
        CPUconvSimpleReverb.h
        main.cpp)

#arm_compute arm_compute_core
target_link_libraries(impl sndfile fftw3 clFFT OpenCL ${Boost_SYSTEM_LIBRARY})
