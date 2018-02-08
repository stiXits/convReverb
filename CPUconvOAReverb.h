#pragma once

#include <cstdint>
#include <fftw3.h>

uint32_t CPUconvOAReverb(float *target, uint32_t targetFrames, float *impulsesx, float *impulsedx, uint32_t impulseFrames, float *outputsx, float *outputdx);
uint32_t padTargetSignal(float* target, uint32_t sampleCount, uint32_t sampleSize, fftw_complex* destinationBuffer);
uint32_t convolve(fftw_complex* targetSignal, fftw_complex* impulseSignal, fftw_complex* transformedSignal,uint32_t sampleSize);
float mergeConvolvedSignal(fftw_complex *longInputBuffer, fftw_complex *shortOutpuBuffer, uint32_t sampleSize, uint32_t sampleCount);
void printComplexArray(fftw_complex *target, uint32_t size);
inline float maximum(float maxo, float value);