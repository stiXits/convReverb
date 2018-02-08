#include "CPUconvOAReverb.h"

#include <math.h>

uint32_t CPUconvOAReverb(float *target, uint32_t targetFrames, float *impulseL, float *impulseR, uint32_t impulseFrames, float *outputL, float *outputR) {

	fftw_plan transformedL_plan_backward, transformedR_plan_backward, impulseL_plan_forward, impulseR_plan_forward;
	uint32_t sampleCount = targetFrames / impulseFrames;
	uint32_t sampleSize = 2 * impulseFrames - 1;
	uint32_t resultSignalSize = sampleSize * sampleCount;

	// FFT buffers
	fftw_complex* impulseSignalL = new fftw_complex[sampleSize];
	fftw_complex* impulseSignalLFT = new fftw_complex[sampleSize];

	fftw_complex* impulseSignalR = new fftw_complex[sampleSize];
	fftw_complex* impulseSignalRFT = new fftw_complex[sampleSize];

	fftw_complex* paddedTargetSignal = new fftw_complex[resultSignalSize];

	fftw_complex* convolvedSignalL = new fftw_complex[resultSignalSize];
	fftw_complex* convolvedSignalR = new fftw_complex[resultSignalSize];

	fftw_complex* mergedSignalL = new fftw_complex[targetFrames + impulseFrames - 1];
	fftw_complex* mergedSignalR = new fftw_complex[targetFrames + impulseFrames - 1];

	padTargetSignal(target, sampleCount, impulseFrames, paddedTargetSignal);

	// copy impulse sound to complex buffer
	for (int i = 0; i < sampleSize; ++i) {
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
	impulseL_plan_forward = fftw_plan_dft_1d(resultSignalSize, impulseSignalL, impulseSignalLFT, FFTW_FORWARD, FFTW_ESTIMATE);
	fftw_execute(impulseL_plan_forward);
	impulseR_plan_forward = fftw_plan_dft_1d(resultSignalSize, impulseSignalR, impulseSignalRFT, FFTW_FORWARD, FFTW_ESTIMATE);
	fftw_execute(impulseR_plan_forward);

	// fourrier transform of target and impulse signal
	// resultSignalSize, targetSignal, targetSignalFt
	for (int i = 0; i < sampleCount; i += sampleSize) {
		// colvolve only parts of the input and output buffers
		convolve(&paddedTargetSignal[i], impulseSignalL, &convolvedSignalL[i], impulseFrames);
		convolve(&paddedTargetSignal[i], impulseSignalR, &convolvedSignalR[i], impulseFrames);
	}

	// backward fourrier transform on transformed signal
	// transformedL_plan_backward = fftw_plan_dft_1d(resultSignalSize, targetSignalsLFT, targetSignalLIFT, FFTW_BACKWARD, FFTW_ESTIMATE);
	// fftw_execute(transformedL_plan_backward);
	// transformedR_plan_backward = fftw_plan_dft_1d(resultSignalSize, targetSignalsRFT, targetSignalRIFT, FFTW_BACKWARD, FFTW_ESTIMATE);
	// fftw_execute(transformedR_plan_backward);

	//printComplexArray(convolvedSignalL, resultSignalSize);

	float maxo[2];
	maxo[0]=0.0f;
	maxo[1]=0.0f; 

	maxo[0] = maximum(maxo[0], mergeConvolvedSignal(convolvedSignalL, mergedSignalL, sampleSize, sampleCount)); 
	maxo[1] = maximum(maxo[1], mergeConvolvedSignal(convolvedSignalR, mergedSignalR, sampleSize, sampleCount)); 

	float maxot= abs(maxo[0])>=abs(maxo[1])? abs(maxo[0]): abs(maxo[1]);

	for (int i=0; i< targetFrames + impulseFrames - 1; i++) {
		//printf("%f\n", targetSignalLIFT[i][0]);
		float temp=0.0f;
		outputL[i]= (float)((mergedSignalL[i][0])/(maxot));
		outputR[i]= (float)((mergedSignalR[i][0])/(maxot));
	}

	// delete [] paddedTargetSignal;
	// delete [] targetSignalsLFT;
	// delete [] targetSignalsRFT;

	// delete [] impulseSignalL;
	// delete [] impulseSignalLFT;

	// delete [] impulseSignalR;
	// delete [] impulseSignalRFT;

	// delete [] targetSignalLIFT;
	// delete [] targetSignalRIFT;

	// delete [] convolvedSignalL;
	// delete [] convolvedSignalR;

	return resultSignalSize;
}

uint32_t convolve(fftw_complex* targetSignal,
	fftw_complex* impulseSignal, 
	fftw_complex* transformedSignal,
	uint32_t sampleSize) {

	// transform signal to frequency domaine
	fftw_plan target_plan_forward = fftw_plan_dft_1d(sampleSize, targetSignal, transformedSignal, FFTW_FORWARD, FFTW_ESTIMATE);	
	fftw_execute(target_plan_forward);

	// convolve target and signal
	for (int i = 0; i < sampleSize; i++) {
		transformedSignal[i][0]= ((impulseSignal[i][0]*transformedSignal[i][1])+(impulseSignal[i][1]*transformedSignal[i][0]));
		transformedSignal[i][1]= ((impulseSignal[i][0]*transformedSignal[i][1])+(impulseSignal[i][1]*transformedSignal[i][0]));
	}

	// transform result back to time domaine
	fftw_plan target_plan_backward = fftw_plan_dft_1d(sampleSize, transformedSignal, targetSignal , FFTW_BACKWARD, FFTW_ESTIMATE);	
	fftw_execute(target_plan_forward);

	// switch buffers
	transformedSignal = impulseSignal;

	return sampleSize;
}

uint32_t padTargetSignal(float* target, uint32_t sampleCount, uint32_t sampleSize, fftw_complex* destinationBuffer) {

	// cut the target signal into samplecount buffers
	uint32_t stride = sampleSize * 2 - 1;

	for (int i = 0; i < sampleCount; ++i) {
		// copy targetsignal into new buffer
		for (int k = 0; k < sampleSize; ++k) {
			int offset = i * stride + k;
			destinationBuffer[offset][0] = target[offset];
			destinationBuffer[offset][1] = 0.0f;
		}

		// pad the buffer with zeros til it reaches a size of samplecount * 2 - 1
		for (int k = 0; k < sampleSize - 1; ++k) {
			int offset = i * stride + sampleSize + k;
			destinationBuffer[offset][0] = 0.0f;
			destinationBuffer[offset][1] = 0.0f;
		}
	}

	return sampleCount * 2 - 1;
}

float mergeConvolvedSignal(fftw_complex *longInputBuffer, fftw_complex *shortOutpuBuffer, uint32_t sampleSize, uint32_t sampleCount) {
	float max = 0; 

	// copy first sample as it is into output buffer
	for (int i = 0; i < sampleSize; ++i) {

		shortOutpuBuffer[i][0] = longInputBuffer[i][0];
		shortOutpuBuffer[i][1] = longInputBuffer[i][1];
		max = maximum(max, longInputBuffer[i][0]);
	}

	uint stride = sampleSize * 2 - 1;
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