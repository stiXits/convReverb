#include "CPUconvSimpleReverb.h"

#include <fftw3.h>
#include <math.h>

int convolve(fftw_complex* targetSignal,
	uint32_t targetFrames, 
	fftw_complex* impulseSignal, 
	uint32_t impulseFrames, 
	fftw_complex* transformedSignal, 
	fftw_complex* transformedSignalTail) {

		int resultSize = targetFrames + impulseFrames - 1;

		fftw_plan target_plan_forward = fftw_plan_dft_1d(resultSize, targetSignal, transformedSignal, FFTW_FORWARD, FFTW_ESTIMATE);	
		fftw_execute(target_plan_forward);

	for (int i=0; i< resultSignalSize; i++){
		transformedSignal[i][0]= ((impulseSignal[i][0]*transformedSignal[i][1])+(impulseSignal[i][1]*transformedSignal[i][0]));
		transformedSignal[i][1]= ((impulseSignal[i][0]*transformedSignal[i][1])+(impulseSignal[i][1]*transformedSignal[i][0]));
	}

	transformedSignalTail = transformedSignal + targetFrames;

	return frames;
}

uint32_t CPUconvOAReverb(float *target, uint32_t targetFrames, float *impulsesx, float *impulsedx, uint32_t impulseFrames, float *outputsx, float *outputdx) {

	fftw_plan transformedSx_plan_backward, transformedDx_plan_backward, impulseSx_plan_forward, impulseDx_plan_forward;
	uint32_t resultSignalSize = targetFrames + impulseFrames - 1;

	// FFT buffers
	fftw_complex* targetSignal = new fftw_complex[resultSignalSize];

	fftw_complex* impulseSignalSx = new fftw_complex[resultSignalSize];
	fftw_complex* impulseSignalSxFt = new fftw_complex[resultSignalSize];

	fftw_complex* impulseSignalDx = new fftw_complex[resultSignalSize];
	fftw_complex* impulseSignalDxFt = new fftw_complex[resultSignalSize];

	fftw_complex* targetSignalSxIft = new fftw_complex[resultSignalSize];
	fftw_complex* targetSignalDxIft = new fftw_complex[resultSignalSize];

	fftw_complex* transformedSignalSx = new fftw_complex[resultSignalSize];
	fftw_complex* transformedSignalDx = new fftw_complex[resultSignalSize];

	// prepare input signal for fft and move it to complex array
	for (int i = 0; i < resultSignalSize; ++i)
	{
		if(i < targetFrames) {
			targetSignal[i][0] = target[i];
		}
		else {
			targetSignal[i][0] = 0.0f;
		}
		targetSignal[i][1] = 0.0f;
	}

	// split up target array into multiple "buffers" of impulseresponse size
	uint32_t sampleCount = targetFrames / impulseFrames;
	fftw_complex** splitUpTargetSignal = new *fftw_complex[sampleCount];

	for (int i = 0; i < count; ++i)
	{
		splitUpTargetSignal[i] = targetSignal + i * impulseFrames;
	}

	for (int i = 0; i < resultSignalSize; ++i)
	{
		if(i < impulseFrames){
			impulseSignalSx[i][0] = impulsesx[i];
			impulseSignalDx[i][0] = impulsedx[i];
		}
		else{
			impulseSignalSx[i][0] = 0.0f;
			impulseSignalDx[i][0] = 0.0f;
		}

		impulseSignalSx[i][1] = 0.0f;
		impulseSignalDx[i][1] = 0.0f;
	}

	// TODO: add Debug mode
	// printf ( "\n#####################################\n\n\n\n\n\n" );
	// printf ( "  Input Data:\n" );
	// printf ( "\n" );

	// for (int i = 0; i < targetFrames; i++ )
	// {
	// 	printf ( "  %3d  %12f  %12f\n", i, targetSignal[i][0], targetSignal[i][1] );
	// }

	impulseSx_plan_forward = fftw_plan_dft_1d(resultSignalSize, impulseSignalSx, impulseSignalSxFt, FFTW_FORWARD, FFTW_ESTIMATE);
	fftw_execute(impulseSx_plan_forward);

	impulseDx_plan_forward = fftw_plan_dft_1d(resultSignalSize, impulseSignalDx, impulseSignalDxFt, FFTW_FORWARD, FFTW_ESTIMATE);
	fftw_execute(impulseDx_plan_forward);



	fftw_complex** targetSignalSFt;

	fftw_complex*  = new fftw_complex[resultSignalSize];
	// fourrier transform of target and impulse signal
	// resultSignalSize, targetSignal, targetSignalFt
	convolve();


	// TODO: add Debug mode
	// printf ( "\n#####################################\n\n\n\n\n\n" );
	// printf ( "  Output FFT Coefficients:\n" );
	// printf ( "\n" );

	// for (int i = 0; i < targetFrames; i++ )
	// {
	// 	printf ( "  %3d  %12f  %12f\n", i, targetSignalFt[i][0], targetSignalFt[i][1] );
	// }

	// backward fourrier transform on transformed signal
	transformedSx_plan_backward = fftw_plan_dft_1d(resultSignalSize, transformedSignalSx, targetSignalSxIft, FFTW_BACKWARD, FFTW_ESTIMATE);
	fftw_execute(transformedSx_plan_backward);
	transformedDx_plan_backward = fftw_plan_dft_1d(resultSignalSize, transformedSignalDx, targetSignalDxIft, FFTW_BACKWARD, FFTW_ESTIMATE);
	fftw_execute(transformedDx_plan_backward);

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
		if (abs(maxo[0])<=abs(targetSignalSxIft[i][0])) maxo[0]=targetSignalSxIft[i][0];
		if (abs(maxo[1])<=abs(targetSignalDxIft[i][0])) maxo[1]=targetSignalDxIft[i][0];
	}
	float maxot= abs(maxo[0])>=abs(maxo[1])? abs(maxo[0]): abs(maxo[1]);

	for (int i=0; i< resultSignalSize; i++){
		float temp=0.0f;
		outputsx[i]= (float)((targetSignalSxIft[i][0])/(maxot));
		outputdx[i]= (float)((targetSignalDxIft[i][0])/(maxot));
	}

	delete [] targetSignal;
	delete [] targetSignalFt;

	delete [] impulseSignalSx;
	delete [] impulseSignalSxFt;

	delete [] impulseSignalDx;
	delete [] impulseSignalDxFt;

	delete [] targetSignalSxIft;
	delete [] targetSignalDxIft;

	delete [] transformedSignalSx;
	delete [] transformedSignalDx;

	return targetFrames;
}