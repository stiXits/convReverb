#pragma once

#include <cstdint>

// including this make fftw3 use std::complex instead of double[2] for complex
// values which is essential for storing them in a vector
#include <complex>
#include <fftw3.h>
#include <vector>


namespace cpuconv {

    uint32_t
    oAReverb(float *target, uint32_t targetFrames, float *impulsesx, float *impulsedx, uint32_t impulseFrames,
             float *outputsx, float *outputdx);

    void padTargetSignal(float *target, uint32_t segmentCount, uint32_t segmentSize, uint32_t transformedSegmentSize,
                             std::vector<fftw_complex> &destinationBuffer);
    void padImpulseSignal(float *impulse, std::vector<fftw_complex> &impulseBuffer, uint32_t  segmentSize, uint32_t transformedSegementSize);

    uint32_t convolve(std::vector<fftw_complex>::iterator targetSignal,
                      std::vector<fftw_complex>::iterator impulseSignal,
                      std::vector<fftw_complex>::iterator intermediateSignal,
                      std::vector<fftw_complex>::iterator transformedSignal,
                      uint32_t sampleSize);

    float mergeConvolvedSignal(std::vector<fftw_complex> &longInputBuffer, std::vector<fftw_complex> &shortOutputBuffer,
                               uint32_t sampleSize, uint32_t sampleCount);

    inline float maximum(float maxo, float value);
}