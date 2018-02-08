#include "CPUconvOAReverb.h"

#include <math.h>
#include <vector>
#include <array>

uint32_t CPUconvOAReverb(float *target, uint32_t targetFrames, float *impulseL, float *impulseR, uint32_t impulseFrames, float *outputL, float *outputR) {

	fftw_plan impulseL_plan_forward, impulseR_plan_forward;
	uint32_t segmentCount = targetFrames / impulseFrames;
  uint32_t segmentSize = impulseFrames;
	uint32_t transformedSegmentSize = 2 * segmentSize - 1;
	uint32_t transformedSignalSize = transformedSegmentSize * segmentCount;

	// FFT buffers
	std::vector<fftw_complex> impulseSignalL(transformedSegmentSize);
  std::vector<fftw_complex> impulseSignalLFT(transformedSegmentSize);

  std::vector<fftw_complex> impulseSignalR(transformedSegmentSize);
  std::vector<fftw_complex> impulseSignalRFT(transformedSegmentSize);

  std::vector<fftw_complex> paddedTargetSignal(transformedSignalSize);

  std::vector<fftw_complex> intermediateSignalL(transformedSignalSize);
  std::vector<fftw_complex> intermediateSignalR(transformedSignalSize);

  std::vector<fftw_complex> convolvedSignalL(transformedSignalSize);
  std::vector<fftw_complex> convolvedSignalR(transformedSignalSize);

  std::vector<fftw_complex> mergedSignalL(transformedSegmentSize * segmentCount);
  std::vector<fftw_complex> mergedSignalR(transformedSegmentSize * segmentCount);

  for (int j = 0; j < transformedSignalSize; ++j) {
    paddedTargetSignal[j][0] = 0.0f;
    paddedTargetSignal[j][1] = 0.0f;
  }

	padTargetSignal(target, segmentCount, segmentSize, paddedTargetSignal);

	// copy impulse sound to complex buffer
	for (int i = 0; i < transformedSegmentSize; ++i) {
		if(i < impulseFrames) {
			impulseSignalL[i][0] = impulseL[i];
			impulseSignalR[i][0] = impulseR[i];
		}
		else {
			impulseSignalL[i][0] = 0.0f;
			impulseSignalR[i][0] = 0.0f;
		}

		impulseSignalL[i][1] = 0.0f;
		impulseSignalR[i][1] = 0.0f;
	}

	// apply fft to impulse l and r
	impulseL_plan_forward = fftw_plan_dft_1d(transformedSegmentSize, impulseSignalL.data(), impulseSignalLFT.data(), FFTW_FORWARD, FFTW_ESTIMATE);
	fftw_execute(impulseL_plan_forward);
	impulseR_plan_forward = fftw_plan_dft_1d(transformedSegmentSize, impulseSignalR.data(), impulseSignalRFT.data(), FFTW_FORWARD, FFTW_ESTIMATE);
	fftw_execute(impulseR_plan_forward);

	// fourrier transform of target and impulse signal
	for (int i = 0; i < segmentCount; i += transformedSegmentSize) {

		// colvolve only parts of the input and output buffers
		convolve(&paddedTargetSignal[i], &impulseSignalL[0], &intermediateSignalL[i], &convolvedSignalL[i], impulseFrames);
		convolve(&paddedTargetSignal[i], &impulseSignalR[0], &intermediateSignalR[i], &convolvedSignalR[i], impulseFrames);
	}

	float maxo[2];
	maxo[0]=0.0f;
	maxo[1]=0.0f; 

	maxo[0] = maximum(maxo[0], mergeConvolvedSignal(convolvedSignalL, mergedSignalL, transformedSegmentSize, segmentCount));
	maxo[1] = maximum(maxo[1], mergeConvolvedSignal(convolvedSignalR, mergedSignalR, transformedSegmentSize, segmentCount));

	float maxot= abs(maxo[0])>=abs(maxo[1])? abs(maxo[0]): abs(maxo[1]);

	for (int i=0; i< targetFrames + impulseFrames - 1; i++) {
		//printf("%f\n", targetSignalLIFT[i][0]);
		float temp=0.0f;
		outputL[i]= (float)((mergedSignalL[i][0])/(maxot));
		outputR[i]= (float)((mergedSignalR[i][0])/(maxot));
	}

	return transformedSignalSize;
}

uint32_t convolve(fftw_complex* targetSignal,
                  fftw_complex* impulseSignal,
                  fftw_complex* intermediateSignal,
                  fftw_complex* transformedSignal,
                  uint32_t sampleSize) {

	// transform signal to frequency domaine
	fftw_plan target_plan_forward = fftw_plan_dft_1d(sampleSize, targetSignal, intermediateSignal, FFTW_FORWARD, FFTW_ESTIMATE);
	fftw_execute(target_plan_forward);

	// convolve target and signal
	for (int i = 0; i < sampleSize; i++) {
    intermediateSignal[i][0]= ((impulseSignal[i][0]*intermediateSignal[i][0])-(impulseSignal[i][1]*intermediateSignal[i][1]));
    intermediateSignal[i][1]= ((impulseSignal[i][0]*intermediateSignal[i][1])+(impulseSignal[i][1]*intermediateSignal[i][0]));
	}

	// transform result back to time domaine
	fftw_plan target_plan_backward = fftw_plan_dft_1d(sampleSize, intermediateSignal, transformedSignal , FFTW_BACKWARD, FFTW_ESTIMATE);
	fftw_execute(target_plan_backward);

	return sampleSize;
}

uint32_t padTargetSignal(float* target, uint32_t segmentCount, uint32_t segmentSize, std::vector<fftw_complex> &destinationBuffer) {

	// cut the target signal into samplecount buffers
	uint32_t stride = segmentSize * 2 - 1;

	for (int i = 0; i < segmentCount; ++i) {
		// copy targetsignal into new buffer
		for (int k = 0; k < segmentSize; ++k) {
			int offset = i * stride + k;
			destinationBuffer[offset][0] = target[offset];
			destinationBuffer[offset][1] = 0.0f;
		}

		// pad the buffer with zeros til it reaches a size of samplecount * 2 - 1
		for (int k = 0; k < segmentSize; ++k) {
			int offset = i * stride + segmentSize + k;
			destinationBuffer[offset][0] = 0.0f;
			destinationBuffer[offset][1] = 0.0f;
		}
	}

	return segmentCount * 2 - 1;
}

float mergeConvolvedSignal(std::vector<fftw_complex> &longInputBuffer, std::vector<fftw_complex> &shortOutpuBuffer, uint32_t sampleSize, uint32_t sampleCount) {
	float max = 0; 

	// copy first sample as it is into output buffer
	for (int i = 0; i < sampleSize; ++i) {

		shortOutpuBuffer[i][0] = longInputBuffer[i][0];
		shortOutpuBuffer[i][1] = longInputBuffer[i][1];
		max = maximum(max, longInputBuffer[i][0]);
	}

  uint32_t currentElement = 0;
	uint32_t stride = sampleSize * 2 - 1;
	// start with second sample, the first one has no signal tail to merge with
	for (int i = 1; i < sampleCount; ++i) {
		uint32_t step = stride * i;

		// signal tail length is sampleSize - 1, so the last element has nothing to be added to
		for (int k = 0; k < sampleSize - 1; ++k) {
			shortOutpuBuffer[step + k][0] = longInputBuffer[step + k][0] + longInputBuffer[step - sampleSize + k][0];
			shortOutpuBuffer[step + k][1] = longInputBuffer[step + k][1] + longInputBuffer[step - sampleSize + k][1];
			max = maximum(max, longInputBuffer[step + k][0]);
		}
		// set last element to the same as in input buffer
		shortOutpuBuffer[step + sampleSize][0] = shortOutpuBuffer[step + sampleSize][0];
		shortOutpuBuffer[step + sampleSize][1] = shortOutpuBuffer[step + sampleSize][1];
		max = maximum(max, longInputBuffer[step + sampleSize][0]);
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
	printf ( "\n#####################################\n\n\n\n\n\n" );
	printf ( "Data (skipping zeros):\n" );
	printf ( "\n" );

	for (int i = 0; i < size; i++ )
	{
		if(target[i][0] != 0.0f || target[i][1] != 0.0f)
		printf ( "  %3d  %12f  %12f\n", i, target[i][0], target[i][1] );
	}
}