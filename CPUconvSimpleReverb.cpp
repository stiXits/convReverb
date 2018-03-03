#include "CPUconvSimpleReverb.h"

#include <fftw3.h>
#include <math.h>

namespace cpuconv {

    uint32_t simpleReverb(float *target, uint32_t targetFrames, float *impulsesx, float *impulsedx,
                          uint32_t impulseFrames, float *outputsx, float *outputdx) {

      fftw_plan target_plan_forward, transformedSx_plan_backward, transformedDx_plan_backward, impulseSx_plan_forward, impulseDx_plan_forward;
      uint32_t resultSignalSize = targetFrames + impulseFrames - 1;

      // FFT buffers
      fftw_complex *targetSignal = new fftw_complex[resultSignalSize];
      fftw_complex *targetSignalFt = new fftw_complex[resultSignalSize];

      fftw_complex *impulseSignalSx = new fftw_complex[resultSignalSize];
      fftw_complex *impulseSignalSxFt = new fftw_complex[resultSignalSize];

      fftw_complex *impulseSignalDx = new fftw_complex[resultSignalSize];
      fftw_complex *impulseSignalDxFt = new fftw_complex[resultSignalSize];

      fftw_complex *targetSignalSxIft = new fftw_complex[resultSignalSize];
      fftw_complex *targetSignalDxIft = new fftw_complex[resultSignalSize];

      fftw_complex *transformedSignalSx = new fftw_complex[resultSignalSize];
      fftw_complex *transformedSignalDx = new fftw_complex[resultSignalSize];

      // prepare input signal for fft and move it to complex array
      for (int i = 0; i < resultSignalSize; ++i) {
        if (i < targetFrames) {
          targetSignal[i][0] = target[i];
        } else {
          targetSignal[i][0] = 0.0f;
        }
        targetSignal[i][1] = 0.0f;
      }

      for (int i = 0; i < resultSignalSize; ++i) {
        if (i < impulseFrames) {
          impulseSignalSx[i][0] = impulsesx[i];
          impulseSignalDx[i][0] = impulsedx[i];
        } else {
          impulseSignalSx[i][0] = 0.0f;
          impulseSignalDx[i][0] = 0.0f;
        }

        impulseSignalSx[i][1] = 0.0f;
        impulseSignalDx[i][1] = 0.0f;
      }

      target_plan_forward = fftw_plan_dft_1d(resultSignalSize, targetSignal, targetSignalFt, FFTW_FORWARD, FFTW_ESTIMATE);
      fftw_execute(target_plan_forward);

      impulseSx_plan_forward = fftw_plan_dft_1d(resultSignalSize, impulseSignalSx, impulseSignalSxFt, FFTW_FORWARD,
                                                FFTW_ESTIMATE);
      fftw_execute(impulseSx_plan_forward);

      impulseDx_plan_forward = fftw_plan_dft_1d(resultSignalSize, impulseSignalDx, impulseSignalDxFt, FFTW_FORWARD,
                                                FFTW_ESTIMATE);
      fftw_execute(impulseDx_plan_forward);

      for (int i = 0; i < resultSignalSize; i++) {
        transformedSignalSx[i][0] = ((impulseSignalSxFt[i][0] * targetSignalFt[i][0]) - (impulseSignalSxFt[i][1] * targetSignalFt[i][1]));
        transformedSignalSx[i][1] = ((impulseSignalSxFt[i][0] * targetSignalFt[i][1]) + (impulseSignalSxFt[i][1] * targetSignalFt[i][0]));
        transformedSignalDx[i][0] = ((impulseSignalDxFt[i][0] * targetSignalFt[i][0]) - (impulseSignalDxFt[i][1] * targetSignalFt[i][1]));
        transformedSignalDx[i][1] = ((impulseSignalDxFt[i][0] * targetSignalFt[i][1]) + (impulseSignalDxFt[i][1] * targetSignalFt[i][0]));
      }

      // backward fourrier transform on transformed signal
      transformedSx_plan_backward = fftw_plan_dft_1d(resultSignalSize, transformedSignalSx, targetSignalSxIft, FFTW_BACKWARD, FFTW_ESTIMATE);
      fftw_execute(transformedSx_plan_backward);

      transformedDx_plan_backward = fftw_plan_dft_1d(resultSignalSize, transformedSignalDx, targetSignalDxIft, FFTW_BACKWARD, FFTW_ESTIMATE);
      fftw_execute(transformedDx_plan_backward);

      float maxo[2];
      maxo[0] = 0.0f;
      maxo[1] = 0.0f;

      for (int i = 0; i < resultSignalSize; i++) {
        if (abs(maxo[0]) <= abs(targetSignalSxIft[i][0])) maxo[0] = targetSignalSxIft[i][0];
        if (abs(maxo[1]) <= abs(targetSignalDxIft[i][0])) maxo[1] = targetSignalDxIft[i][0];
      }
      float maxot = abs(maxo[0]) >= abs(maxo[1]) ? abs(maxo[0]) : abs(maxo[1]);

      for (int i = 0; i < resultSignalSize; i++) {
        float temp = 0.0f;
        outputsx[i] = (float) ((targetSignalSxIft[i][0]) / (maxot));
        outputdx[i] = (float) ((targetSignalDxIft[i][0]) / (maxot));
      }

      delete[] targetSignal;
      delete[] targetSignalFt;

      delete[] impulseSignalSx;
      delete[] impulseSignalSxFt;

      delete[] impulseSignalDx;
      delete[] impulseSignalDxFt;

      delete[] targetSignalSxIft;
      delete[] targetSignalDxIft;

      delete[] transformedSignalSx;
      delete[] transformedSignalDx;

      return resultSignalSize;
    }
}