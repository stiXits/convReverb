#include "CPUconvOAReverb.h"

#include <math.h>
#include <array>
#include <ctime>

namespace cpuconv {

    uint32_t
    oAReverb(float* &target, uint32_t targetFrames, float* &impulseL, float* &impulseR, uint32_t impulseFrames,
             float *outputL, float *outputR) {

      fftw_plan impulseL_plan_forward, impulseR_plan_forward;
      uint32_t segmentCount = targetFrames / impulseFrames;
      if(segmentCount < 1) {
        segmentCount = 1;
      }
      uint32_t segmentSize = impulseFrames;
      uint32_t transformedSegmentSize = 2 * segmentSize - 1;
      uint32_t transformedSignalSize = (transformedSegmentSize) * segmentCount;

      std::vector<fftw_complex> impulseSignalL(transformedSegmentSize);
      std::vector<fftw_complex> impulseSignalLFT(transformedSegmentSize);

      std::vector<fftw_complex> impulseSignalR(transformedSegmentSize);
      std::vector<fftw_complex> impulseSignalRFT(transformedSegmentSize);

      std::vector<fftw_complex> paddedTargetSignal(transformedSignalSize);

      std::vector<fftw_complex> intermediateSignalL(transformedSignalSize);
      std::vector<fftw_complex> intermediateSignalR(transformedSignalSize);

      std::vector<fftw_complex> convolvedSignalL(transformedSignalSize);
      std::vector<fftw_complex> convolvedSignalR(transformedSignalSize);

      // the resultsignal is impulsesize longer than the original
      std::vector<fftw_complex> mergedSignalL(segmentSize * (segmentCount + 1));
      std::vector<fftw_complex> mergedSignalR(segmentSize * (segmentCount + 1));

      padTargetSignal(target, segmentCount, segmentSize, transformedSegmentSize, paddedTargetSignal);

      padImpulseSignal(impulseL, impulseSignalL, segmentSize, transformedSegmentSize);
      padImpulseSignal(impulseR, impulseSignalR, segmentSize, transformedSegmentSize);

      // apply fft to impulse l and r
      impulseL_plan_forward = fftw_plan_dft_1d(transformedSegmentSize, impulseSignalL.data(), impulseSignalLFT.data(),
                                               FFTW_FORWARD, FFTW_ESTIMATE);
      fftw_execute(impulseL_plan_forward);
      impulseR_plan_forward = fftw_plan_dft_1d(transformedSegmentSize, impulseSignalR.data(), impulseSignalRFT.data(),
                                               FFTW_FORWARD, FFTW_ESTIMATE);
      fftw_execute(impulseR_plan_forward);

      clock_t begin = clock();
      // fourrier transform of target and impulse signal
      for (int i = 0; i < transformedSignalSize; i += transformedSegmentSize) {

        // chnlvolve only parts of the input and output buffers
        convolve(paddedTargetSignal.begin() + i, impulseSignalLFT.begin(), intermediateSignalL.begin() + i, convolvedSignalL.begin() + i,
                 transformedSegmentSize);
        convolve(paddedTargetSignal.begin() + i, impulseSignalRFT.begin(), intermediateSignalR.begin() + i, convolvedSignalR.begin() + i,
                 transformedSegmentSize);
      }

      float maxo[2];
      maxo[0] = 0.0f;
      maxo[1] = 0.0f;

      maxo[0] = maximum(maxo[0], mergeConvolvedSignal(convolvedSignalL, mergedSignalL, segmentSize, segmentCount));
      maxo[1] = maximum(maxo[1], mergeConvolvedSignal(convolvedSignalR, mergedSignalR, segmentSize, segmentCount));

      clock_t end = clock();
      double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
      printf("time consumed for raw convolution: %f seconds\n", elapsed_secs);

      float maxot = abs(maxo[0]) >= abs(maxo[1]) ? abs(maxo[0]) : abs(maxo[1]);

      for (int i = 0; i < segmentSize * (segmentCount + 1); i++) {
        outputL[i] = (float) ((mergedSignalL[i][0]) / (maxot));
        outputR[i] = (float) ((mergedSignalR[i][0]) / (maxot));
      }

      return segmentSize * (segmentCount + 1);
    }

    uint32_t convolve(std::vector<fftw_complex>::iterator targetSignal,
                      std::vector<fftw_complex>::iterator impulseSignal,
                      std::vector<fftw_complex>::iterator intermediateSignal,
                      std::vector<fftw_complex>::iterator transformedSignal,
                      uint32_t sampleSize) {

      std::vector<fftw_complex >::iterator cachedIntermediateSignalStart = intermediateSignal;
      // transform signal to frequency domaine
      fftw_plan target_plan_forward = fftw_plan_dft_1d(sampleSize, &*targetSignal, &*intermediateSignal, FFTW_FORWARD, FFTW_ESTIMATE);
      fftw_execute(target_plan_forward);

      for (int i = 0; i < sampleSize; i++) {
        float cacheReal = ((*impulseSignal)[0] * (*intermediateSignal)[0] - (*impulseSignal)[1] * (*intermediateSignal)[1]);
        float cacheImaginary = ((*impulseSignal)[0] * (*intermediateSignal)[1] + (*impulseSignal)[1] * (*intermediateSignal)[0]);
        (*intermediateSignal)[0] = cacheReal;
        (*intermediateSignal)[1] = cacheImaginary;

        intermediateSignal++;
        impulseSignal++;
      }

      // transform result back to time domaine
      fftw_plan target_plan_backward = fftw_plan_dft_1d(sampleSize, &*cachedIntermediateSignalStart, &*transformedSignal, FFTW_BACKWARD, FFTW_ESTIMATE);
      fftw_execute(target_plan_backward);

      return sampleSize;
    }

    void padImpulseSignal(float *impulse, std::vector<fftw_complex> &impulseBuffer, uint32_t  segmentSize, uint32_t transformedSegmentSize)
    {
      // copy impulse sound to complex buffer
      for (int i = 0; i < transformedSegmentSize ; i++) {
        if (i < segmentSize) {
          impulseBuffer[i][0] = impulse[i];
        } else {
          impulseBuffer[i][0] = 0.0f;
        }

        impulseBuffer[i][1] = 0.0f;
      }
    }

    void padTargetSignal(float *target, uint32_t segmentCount, uint32_t segmentSize, uint32_t transformedSegmentSize,
                             std::vector<fftw_complex> &destinationBuffer) {

      for (int i = 0; i < segmentCount; ++i) {
        // copy targetsignal into new buffer
        for (int k = 0; k < segmentSize; ++k) {
          int readOffset = segmentSize * i + k;
          int writeOffset = transformedSegmentSize * i + k;

          destinationBuffer[writeOffset][0] = target[readOffset];
          destinationBuffer[writeOffset][1] = 0.0f;
        }

        // pad the buffer with zeros til it reaches a size of samplesize * 2 - 1
        for (int k = 0; k < transformedSegmentSize - segmentSize; ++k) {
          int writeOffset = transformedSegmentSize + segmentSize +  k;
          destinationBuffer[writeOffset][0] = 0.0f;
          destinationBuffer[writeOffset][1] = 0.0f;
        }
      }
    }

    float mergeConvolvedSignal(std::vector<fftw_complex> &longInputBuffer, std::vector<fftw_complex> &shortOutputBuffer,
                               uint32_t sampleSize, uint32_t sampleCount) {
      float max = 0;
      uint32_t stride = sampleSize * 2;
      // start with second sample, the first one has no signal tail to merge with
      for (int i = 0; i <= sampleCount; ++i) {
        uint32_t readHeadPosition = stride * i;
        // tail has length samplesize - 1 so the resulting + 1
        uint32_t readTailPosition = readHeadPosition - sampleSize;
        uint32_t writePosition = sampleSize * i;

        for (int k = 0; k < sampleSize; k++) {
          if (i == 0) {
            // position is in an area where no tail exists, yet. Speaking the very first element:
            shortOutputBuffer[writePosition + k][0] = (float)(longInputBuffer[readHeadPosition + k][0]);
            shortOutputBuffer[writePosition + k][1] = (float)(longInputBuffer[readHeadPosition + k][1]);
            max = maximum(max, shortOutputBuffer[writePosition + k][0]);
          }
          else if (i == sampleCount) {
            // segment add the last tail to output
            shortOutputBuffer[writePosition + k][0] = (float)(longInputBuffer[readTailPosition + k][0]);
            shortOutputBuffer[writePosition + k][1] = (float)(longInputBuffer[readTailPosition + k][1]);
            max = maximum(max, shortOutputBuffer[writePosition + k][0]);
          } else {
            // segment having a head and a tail to summ up
            shortOutputBuffer[writePosition + k][0] =
                    (float)(longInputBuffer[readHeadPosition + k][0] + longInputBuffer[readTailPosition + k][0]);
            shortOutputBuffer[writePosition + k + 1][0] =
                    (float)(longInputBuffer[readHeadPosition + k][1] + longInputBuffer[readTailPosition + k][1]);
            max = maximum(max, shortOutputBuffer[writePosition + k][0]);
          }
        }
      }

      return max;
    }

    inline float maximum(float max, float value) {
      if (abs(max) <= abs(value)) {
        max = value;
      }

      return max;
    }
}