#pragma once

#include <cstdint>

// including this make fftw3 use std::complex instead of double[2] for complex
// values which is essential for storing them in a vector
#include <complex>
#include <vector>
#include <clFFT.h>
#include <boost/iterator/zip_iterator.hpp>
#include <boost/range/iterator_range_core.hpp>
#include <boost/iterator.hpp>

namespace gpuconv {

    typedef std::complex<float> complex;

    // wrappers for CL specific operations
    void setUpCL();
    cl_mem createGPUBuffer(std::vector<complex>::iterator &buffer, uint32_t bufferSize);
    void enqueueGPUWriteBuffer(cl_mem &bufferHandle, std::vector<complex>::iterator &buffer, uint32_t bufferSize);
    void enqueueGPUReadBuffer(cl_mem &bufferHandle, std::vector<complex>::iterator &buffer, uint32_t bufferSize);
    clfftPlanHandle createGPUPlan(uint32_t bufferSize);
    void enqueueGPUPlan(clfftPlanHandle planHandle, cl_mem &bufferHandle, clfftDirection direction);

    void fftSingle(std::vector<complex>::iterator buffer, uint32_t bufferSize, clfftDirection direction,
                   cl_command_queue queue);
    void fftParallel(std::vector<std::vector<complex>::iterator> buffers, uint32_t bufferSize,
                     clfftDirection direction,
                     cl_command_queue queue);

    uint32_t
    oAReverb(float *target, uint32_t targetFrames, float *impulsesx, float *impulsedx, uint32_t impulseFrames,
             float *outputsx, float *outputdx);

    void padTargetSignal(float *target, uint32_t segmentCount, uint32_t segmentSize, uint32_t transformedSegmentSize,
                             std::vector<complex> &destinationBuffer);
    void padImpulseSignal(float *impulse, std::vector<complex> &impulseBuffer, uint32_t  segmentSize, uint32_t transformedSegementSize);

    uint32_t convolve(std::vector<complex>::iterator targetSignal,
                      std::vector<complex>::iterator impulseSignal,
                      uint32_t bufferSize);
    uint32_t convolveParallel(std::vector<std::vector<complex>::iterator> targetSignals,
                              std::vector<complex>::iterator impulseSignal,
                              uint32_t bufferSize);

    float mergeConvolvedSignal(std::vector<complex> &longInputBuffer, std::vector<complex> &shortOutputBuffer,
                               uint32_t sampleSize, uint32_t sampleCount);

    inline float maximum(float maxo, float value);

    template<class... Conts>
    auto zip_range(Conts&... conts)
    -> decltype(boost::make_iterator_range(
            boost::make_zip_iterator(boost::make_tuple(conts.begin()...)),
            boost::make_zip_iterator(boost::make_tuple(conts.end()...))))
    {
      return {boost::make_zip_iterator(boost::make_tuple(conts.begin()...)),
              boost::make_zip_iterator(boost::make_tuple(conts.end()...))};
    }
}