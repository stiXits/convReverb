#pragma once

#include <cstdint>

uint32_t CPUconvSimpleReverb(float *target, uint32_t targetFrames, float *impulsesx, float *impulsedx, uint32_t impulseFrames, float *outputsx, float *outputdx);