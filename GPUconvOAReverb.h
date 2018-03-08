#pragma once

#include <cstdint>

// including this make fftw3 use std::complex instead of double[2] for complex
// values which is essential for storing them in a vector
#include <complex>
#include <vector>
#include <clFFT.h>

namespace gpuconv {

    typedef float fftw_complex[2];

    void setUpCL();
    void fft(std::vector<fftw_complex>::iterator buffer, uint32_t bufferSize, clfftDirection direction, cl_command_queue queue, cl_context ctx);

    uint32_t
    oAReverb(float *target, uint32_t targetFrames, float *impulsesx, float *impulsedx, uint32_t impulseFrames,
             float *outputsx, float *outputdx);

    void padTargetSignal(float *target, uint32_t segmentCount, uint32_t segmentSize, uint32_t transformedSegmentSize,
                             std::vector<fftw_complex> &destinationBuffer);
    void padImpulseSignal(float *impulse, std::vector<fftw_complex> &impulseBuffer, uint32_t  segmentSize, uint32_t transformedSegementSize);

    uint32_t convolve(std::vector<fftw_complex>::iterator targetSignal,
                      std::vector<fftw_complex>::iterator impulseSignal,
                      uint32_t sampleSize);

    float mergeConvolvedSignal(std::vector<fftw_complex> &longInputBuffer, std::vector<fftw_complex> &shortOutputBuffer,
                               uint32_t sampleSize, uint32_t sampleCount);

    void printComplexArray(fftw_complex *target, uint32_t size);

    inline float maximum(float maxo, float value);

    uint32_t transform(std::vector<fftw_complex> target,
                       std::vector<fftw_complex> impulse,
                       uint32_t sampleSize,
                       uint32_t segmentCount,
                       cl_command_queue queue,
                       cl_context);
}