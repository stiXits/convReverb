#include "CPUconvOAReverb.h"

#include <math.h>

uint32_t CPUconvOAReverb(float *target, uint32_t targetFrames, float *impulseL, float *impulseR, uint32_t impulseFrames, float *outputsx, float *outputdx) {

	fftw_plan transformedL_plan_backward, transformedR_plan_backward, impulseL_plan_forward, impulseR_plan_forward;
	uint32_t sampleCount = targetFrames / impulseFrames;
	uint32_t sampleSize = 2 * impulseFrames - 1;
	uint32_t resultSignalSize = sampleSize * sampleCount;

	// FFT buffers
	fftw_complex* impulseSignalL = new fftw_complex[sampleSize];
	fftw_complex* impulseSignalLFt = new fftw_complex[sampleSize];

	fftw_complex* impulseSignalR = new fftw_complex[sampleSize];
	fftw_complex* impulseSignalRFt = new fftw_complex[sampleSize];

	fftw_complex* targetSignalLIft = new fftw_complex[resultSignalSize];
	fftw_complex* targetSignalRIft = new fftw_complex[resultSignalSize];

	fftw_complex* transformedSignalL = new fftw_complex[resultSignalSize];
	fftw_complex* transformedSignalR = new fftw_complex[resultSignalSize];

	fftw_complex* paddedTargetSignal = new fftw_complex[resultSignalSize];
	fftw_complex* targetSignalSFt = new fftw_complex[resultSignalSize];

	padTargetSignal(target, sampleCount, impulseFrames, paddedTargetSignal);

	// copy impulse sound to complex buffer
	for (int i = 0; i < sampleSize; ++i)
	{
		if(i < impulseFrames){
			impulseSignalL[i][0] = impulseL[i];
			impulseSignalR[i][0] = impulseR[i];
		}
		else{
			impulseSignalL[i][0] = 0.0f;
			impulseSignalR[i][0] = 0.0f;
		}

		impulseSignalL[i][1] = 0.0f;
		impulseSignalR[i][1] = 0.0f;
	}

	// TODO: add Debug mode
	// printf ( "\n#####################################\n\n\n\n\n\n" );
	// printf ( "  Input Data:\n" );
	// printf ( "\n" );

	// for (int i = 0; i < targetFrames; i++ )
	// {
	// 	printf ( "  %3d  %12f  %12f\n", i, targetSignal[i][0], targetSignal[i][1] );
	// }

	// apply fft to impulse l and r
	impulseL_plan_forward = fftw_plan_dft_1d(resultSignalSize, impulseSignalL, impulseSignalLFt, FFTW_FORWARD, FFTW_ESTIMATE);
	fftw_execute(impulseL_plan_forward);
	impulseR_plan_forward = fftw_plan_dft_1d(resultSignalSize, impulseSignalR, impulseSignalRFt, FFTW_FORWARD, FFTW_ESTIMATE);
	fftw_execute(impulseR_plan_forward);




	// fourrier transform of target and impulse signal
	// resultSignalSize, targetSignal, targetSignalFt
	for (int i = 0; i < sampleCount; i += sampleSize)
	{
		// colvolve only parts of the input and output buffers
		convolve(&paddedTargetSignal[i], impulseSignalR, &targetSignalSFt[i], sampleSize);
	}

	// TODO: add Debug mode
	// printf ( "\n#####################################\n\n\n\n\n\n" );
	// printf ( "  Output FFT Coefficients:\n" );
	// printf ( "\n" );

	// for (int i = 0; i < targetFrames; i++ )
	// {
	// 	printf ( "  %3d  %12f  %12f\n", i, targetSignalFt[i][0], targetSignalFt[i][1] );
	// }

	// backward fourrier transform on transformed signal
	transformedL_plan_backward = fftw_plan_dft_1d(resultSignalSize, transformedSignalL, targetSignalLIft, FFTW_BACKWARD, FFTW_ESTIMATE);
	fftw_execute(transformedL_plan_backward);
	transformedR_plan_backward = fftw_plan_dft_1d(resultSignalSize, transformedSignalR, targetSignalRIft, FFTW_BACKWARD, FFTW_ESTIMATE);
	fftw_execute(transformedR_plan_backward);

	// TODO: add Debug mode
	// printf ( "\n" );
	// printf ( "\n#####################################\n\n\n\n\n\n" );
	// printf ( "\n" );

	// for (int i = 0; i < targetFrames; i++ )
	// {
	// printf ( "  %3d  %12f  %12f\n", i, 
	// 	targetSignalIft[i][0] / ( double ) ( targetFrames ), targetSignalIft[i][1] / ( double ) ( targetFrames ) );
	// }

	float maxo[2];
	maxo[0]=0.0f;
	maxo[1]=0.0f;  

	for (int i = 0; i < resultSignalSize; i++){
		if (abs(maxo[0])<=abs(targetSignalLIft[i][0])) maxo[0]=targetSignalLIft[i][0];
		if (abs(maxo[1])<=abs(targetSignalRIft[i][0])) maxo[1]=targetSignalRIft[i][0];
	}
	float maxot= abs(maxo[0])>=abs(maxo[1])? abs(maxo[0]): abs(maxo[1]);

	for (int i=0; i< resultSignalSize; i++){
		float temp=0.0f;
		outputsx[i]= (float)((targetSignalLIft[i][0])/(maxot));
		outputdx[i]= (float)((targetSignalRIft[i][0])/(maxot));
	}

	delete [] paddedTargetSignal;
	delete [] targetSignalSFt;

	delete [] impulseSignalL;
	delete [] impulseSignalLFt;

	delete [] impulseSignalR;
	delete [] impulseSignalRFt;

	delete [] targetSignalLIft;
	delete [] targetSignalRIft;

	delete [] transformedSignalL;
	delete [] transformedSignalR;

	return targetFrames;
}

uint32_t convolve(fftw_complex* targetSignal,
	fftw_complex* impulseSignal, 
	fftw_complex* transformedSignal,
	uint32_t sampleSize) {

	fftw_plan target_plan_forward = fftw_plan_dft_1d(sampleSize, targetSignal, transformedSignal, FFTW_FORWARD, FFTW_ESTIMATE);	
	fftw_execute(target_plan_forward);

	for (int i = 0; i < sampleSize; i++){
		transformedSignal[i][0]= ((impulseSignal[i][0]*transformedSignal[i][1])+(impulseSignal[i][1]*transformedSignal[i][0]));
		transformedSignal[i][1]= ((impulseSignal[i][0]*transformedSignal[i][1])+(impulseSignal[i][1]*transformedSignal[i][0]));
	}

	return sampleSize;
}

uint32_t padTargetSignal(float* target, uint32_t sampleCount, uint32_t sampleSize, fftw_complex* destinationBuffer) {

	// cut the target signal into samplecount buffers
	uint32_t stride = sampleSize * 2 - 1;

	for (int i = 0; i < sampleCount; ++i)
	{
		// copy targetsignal into new buffer
		for (int k = 0; k < sampleSize; ++k)
		{
			int offset = i * stride + k;
			destinationBuffer[offset][0] = target[offset];
			destinationBuffer[offset][1] = 0.0f;
		}

		// pad the buffer with zeros til it reaches a size of samplecount * 2 - 1
		for (int k = 0; k < sampleSize - 1; ++k)
		{
			int offset = i * stride + sampleSize + k;
			destinationBuffer[offset][0] = 0.0f;
			destinationBuffer[offset][1] = 0.0f;
		}
	}

	return sampleCount * 2 - 1;
}