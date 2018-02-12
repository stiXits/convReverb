#pragma once

#include <cstdint>

// including this make fftw3 use std::complex instead of double[2] for complex
// values which is essential for storing them in a vector
#include <complex>
#include <fftw3.h>
#include <vector>
#include <clFFT.h>

namespace gpuconv {

    void setUpCL(uint32_t bufferSize);
    void fft(float *buffer, cl_mem bufferHandle, uint32_t bufferSize, clfftDirection direction, cl_command_queue queue, cl_context ctx);

    uint32_t
    oAReverb(float *target, uint32_t targetFrames, float *impulsesx, float *impulsedx, uint32_t impulseFrames,
             float *outputsx, float *outputdx);

    uint32_t padTargetSignal(float *target, uint32_t segmentCount, uint32_t segmentSize,
                             float *destinationBuffer);
    void padImpulseSignal(float *impulse, float *impulseBuffer, uint32_t segmentSize);

    uint32_t transform(float *target,
                       float *impulse,
                       cl_mem impulseBuffer,
                       uint32_t sampleSize,
                       uint32_t segmentCount,
                       cl_command_queue queue,
                       cl_context);

    uint32_t convolve(float *targetSignal,
                      cl_mem targetBuffer,
                      float *impulseSignal,
                      cl_mem impulseBuffer,
                      uint32_t sampleSize,
                      cl_command_queue queue,
                      cl_context);

    float mergeConvolvedSignal(std::vector<fftw_complex> &longInputBuffer, std::vector<fftw_complex> &shortOutpuBuffer,
                               uint32_t sampleSize, uint32_t sampleCount);

    void printComplexArray(fftw_complex *target, uint32_t size);

    inline float maximum(float maxo, float value);
}