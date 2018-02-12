#pragma once

#include <cstdint>

namespace cpuconv {
    uint32_t simpleReverb(float *target, uint32_t targetFrames, float *impulsesx, float *impulsedx,
                          uint32_t impulseFrames, float *outputsx, float *outputdx);
}