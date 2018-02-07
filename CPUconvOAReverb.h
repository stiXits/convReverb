#pragma once

#include <cstdint>
#include <fftw3.h>

uint32_t CPUconvOAReverb(float *target, uint32_t targetFrames, float *impulsesx, float *impulsedx, uint32_t impulseFrames, float *outputsx, float *outputdx);
uint32_t padTargetSignal(float* target, uint32_t sampleCount, uint32_t sampleSize, fftw_complex* destinationBuffer);
uint32_t convolve(fftw_complex* targetSignal,
	uint32_t targetFrames, 
	fftw_complex* impulseSignal, 
	uint32_t impulseFrames, 
	fftw_complex* transformedSignal);