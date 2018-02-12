#pragma once

#include <cstdint>

namespace cpuconv {
    uint32_t identity(float *target, uint32_t targetFrames, float *outputsx, float *outputdx);
}