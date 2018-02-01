#pragma once

#include <cstdint>

uint32_t CPUconvIdentitySimpleReverb(float *target, uint32_t targetFrames, float *impulsesx, float *impulsedx, uint32_t impulseFrames, float *outputsx, float *outputdx);