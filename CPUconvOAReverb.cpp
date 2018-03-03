#include "CPUconvOAReverb.h"

#include <math.h>
#include <array>

namespace cpuconv {

// FFT buffers
    std::vector<fftw_complex> impulseSignalL;
    std::vector<fftw_complex> impulseSignalLFT;

    std::vector<fftw_complex> impulseSignalR;
    std::vector<fftw_complex> impulseSignalRFT;

    std::vector<fftw_complex> paddedTargetSignal;

    std::vector<fftw_complex> intermediateSignalL;
    std::vector<fftw_complex> intermediateSignalR;

    std::vector<fftw_complex> convolvedSignalL;
    std::vector<fftw_complex> convolvedSignalR;

    std::vector<fftw_complex> mergedSignalL;
    std::vector<fftw_complex> mergedSignalR;

    uint32_t
    oAReverb(float *target, uint32_t targetFrames, float *impulseL, float *impulseR, uint32_t impulseFrames,
             float *outputL, float *outputR) {

      fftw_plan impulseL_plan_forward, impulseR_plan_forward;
      uint32_t segmentCount = targetFrames / impulseFrames;
      uint32_t segmentSize = impulseFrames;
      uint32_t transformedSegmentSize = 2 * segmentSize - 1;
      uint32_t transformedSignalSize = (transformedSegmentSize) * segmentCount;

      impulseSignalL = std::vector<fftw_complex>(transformedSegmentSize);
      impulseSignalLFT = std::vector<fftw_complex>(transformedSegmentSize);

      impulseSignalR = std::vector<fftw_complex>(transformedSegmentSize);
      impulseSignalRFT = std::vector<fftw_complex>(transformedSegmentSize);

      paddedTargetSignal = std::vector<fftw_complex>(transformedSignalSize);

      intermediateSignalL = std::vector<fftw_complex>(transformedSignalSize);
      intermediateSignalR = std::vector<fftw_complex>(transformedSignalSize);

      convolvedSignalL = std::vector<fftw_complex>(transformedSignalSize);
      convolvedSignalR = std::vector<fftw_complex>(transformedSignalSize);

      // the resultsignal is impulsesize longer than the original
      mergedSignalL = std::vector<fftw_complex>(segmentSize * (segmentCount + 1));
      mergedSignalR = std::vector<fftw_complex>(segmentSize * (segmentCount + 1));

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

//      maxo[0] = maximum(maxo[0], mergeConvolvedSignal(convolvedSignalL, mergedSignalL, segmentSize, segmentCount));
//      maxo[1] = maximum(maxo[1], mergeConvolvedSignal(convolvedSignalR, mergedSignalR, segmentSize, segmentCount));
//
//      maxo[0] = maximum(maxo[0], mergeConvolvedSignal(impulseSignalL, mergedSignalL, segmentSize, segmentCount));
//      maxo[1] = maximum(maxo[1], mergeConvolvedSignal(impulseSignalR, mergedSignalR, segmentSize, segmentCount));

      for (int j = 0; j < transformedSignalSize; j++) {
        maxo[0] = maximum(maxo[0], paddedTargetSignal[j][0]);
        maxo[1] = maximum(maxo[1], paddedTargetSignal[j][0]);
      }

      float maxot = abs(maxo[0]) >= abs(maxo[1]) ? abs(maxo[0]) : abs(maxo[1]);

      for (int i = 0; i < transformedSignalSize; i++) {
        outputL[i] = (float) ((paddedTargetSignal[i][0]) / (maxot));
        outputR[i] = (float) ((paddedTargetSignal[i][0]) / (maxot));
      }

//      for (int i = 0; i < targetFrames + impulseFrames - 1; i++) {
//        outputL[i] = (float) ((mergedSignalL[i][0]) / (maxot));
//        outputR[i] = (float) ((mergedSignalR[i][0]) / (maxot));
//      }

      return transformedSignalSize;
    }

    uint32_t convolve(std::vector<fftw_complex>::iterator targetSignal,
                      std::vector<fftw_complex>::iterator impulseSignal,
                      std::vector<fftw_complex>::iterator intermediateSignal,
                      std::vector<fftw_complex>::iterator transformedSignal,
                      uint32_t sampleSize) {

      std::vector<fftw_complex >::iterator cachedIntermediateSignalStart = intermediateSignal;
      // transform signal to frequency domaine
      fftw_plan target_plan_forward = fftw_plan_dft_1d(sampleSize, &*targetSignal, &*intermediateSignal, FFTW_FORWARD,
                                                       FFTW_ESTIMATE);
      fftw_execute(target_plan_forward);

      // convolve target and signal
//      for (int i = 0; i < sampleSize; i++) {
//        intermediateSignal[i][0] = intermediateSignal[i][0];
//        intermediateSignal[i][1] = intermediateSignal[i][1];
//      }

      for (int i = 0; i < sampleSize; i++) {
        float cacheReal = (*intermediateSignal)[0];
        float cacheImaginary =  (*intermediateSignal)[1];
        (*intermediateSignal)[0] = ((*impulseSignal)[0] * cacheReal - (*impulseSignal)[1] * cacheImaginary);
        (*intermediateSignal)[1] = ((*impulseSignal)[0] * cacheImaginary + (*impulseSignal)[1] * cacheReal);

        intermediateSignal++;
        impulseSignal++;
      }

      // transform result back to time domaine
      fftw_plan target_plan_backward = fftw_plan_dft_1d(sampleSize, &*cachedIntermediateSignalStart, &*transformedSignal,
                                                        FFTW_BACKWARD, FFTW_ESTIMATE);
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
          int writeOffset = segmentSize * 2 * i + k;

          destinationBuffer[writeOffset][0] = target[readOffset];
          destinationBuffer[writeOffset][1] = 0.0f;
        }

        // pad the buffer with zeros til it reaches a size of samplesize * 2 - 1
        for (int k = 0; k < transformedSegmentSize - segmentSize; ++k) {
          int writeOffset = segmentSize * i * 2 + segmentSize +  k;
          destinationBuffer[writeOffset][0] = 1.0f;
          destinationBuffer[writeOffset][1] = 0.0f;
        }
      }
    }

    float mergeConvolvedSignal(std::vector<fftw_complex> &longInputBuffer, std::vector<fftw_complex> &shortOutpuBuffer,
                               uint32_t sampleSize, uint32_t sampleCount) {
      float max = 0;
      uint32_t stride = sampleSize * 2;
      // start with second sample, the first one has no signal tail to merge with
      for (int i = 0; i <= sampleCount; ++i) {
        uint32_t readHeadPosition = stride * i;
        // tail has length samplesize - 1 so the resulting + 1
        uint32_t readTailPosition = readHeadPosition - sampleSize;
        uint32_t writePosition = sampleSize * i;

        for (int k = 0; k < sampleSize * 2; k += 2) {
          if (i == 0) {
            // position is in an area where no tail exists, yet. Speaking the very first element:
            shortOutpuBuffer[writePosition + k][0] = longInputBuffer[readHeadPosition + k][0];
            shortOutpuBuffer[writePosition + k][1] = longInputBuffer[readHeadPosition + k][1];
            max = maximum(max, shortOutpuBuffer[writePosition + k][0]);
          }
          else if (i == sampleCount) {
            // segment add the last tail to output
            shortOutpuBuffer[writePosition + k][0] = longInputBuffer[readTailPosition + k][0];
            shortOutpuBuffer[writePosition + k][1] = longInputBuffer[readTailPosition + k][1];
            max = maximum(max, shortOutpuBuffer[writePosition + k][0]);
          } else {
            // segment having a head and a tail to summ up
            shortOutpuBuffer[writePosition + k][0] =
                    longInputBuffer[readHeadPosition + k][0] + longInputBuffer[readTailPosition + k][0];
            shortOutpuBuffer[writePosition + k + 1][0] =
                    longInputBuffer[readHeadPosition + k][1] + longInputBuffer[readTailPosition + k][1];
            max = maximum(max, shortOutpuBuffer[writePosition + k][0]);
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

    void printComplexArray(fftw_complex *target, uint32_t size) {
      printf("\n#####################################\n\n\n\n\n\n");
      printf("Data (skipping zeros):\n");
      printf("\n");

      for (int i = 0; i < size; i++) {
        if (target[i][0] != 0.0f || target[i][1] != 0.0f)
          printf("  %3d  %12f  %12f\n", i, target[i][0], target[i][1]);
      }
    }

    void compareVectors(std::vector<fftw_complex> vec0, std::vector<fftw_complex> vec1, uint32_t size) {
      for (int i = 0; i < size; ++i) {
        if (vec0[0] != vec1[0] || vec0[1] != vec1[1]) {
          printf("Differing vectors:\n");
          printf("%12f  %12f\n", i, vec0[0], vec1[1]);
          printf("%12f  %12f\n", i, vec0[0], vec1[1]);
        }
      }
    }
}