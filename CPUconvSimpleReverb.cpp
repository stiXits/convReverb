#include "CPUconvSimpleReverb.h"

#include <fftw3.h>
#include <math.h>

uint32_t CPUconvSimpleReverb(float *target, uint32_t targetFrames, float *impulsesx, float *impulsedx, uint32_t impulseFrames, float *outputsx, float *outputdx) {

	fftw_plan target_plan_forward, target_plan_backward, impulse_plan_forward;

	// FFT buffers
	fftw_complex* targetSignal = new fftw_complex[targetFrames];
	fftw_complex* targetSignalFt = new fftw_complex[targetFrames];

	fftw_complex* impulseSignal = new fftw_complex[targetFrames];
	fftw_complex* impulseSignalFt = new fftw_complex[targetFrames];

	fftw_complex* targetSignalIft = new fftw_complex[targetFrames];

	// prepare input signal for fft and move it to complex array
	for (int i = 0; i < targetFrames; ++i)
	{
		targetSignal[i][0] = target[i];
		targetSignal[i][1] = 0.0f;
	}

	for (int i = 0; i < impulseFrames; ++i)
	{
		impulseSignal[i][0] = impulsesx[i];
		impulseSignal[i][1] = 0.0f;
	}

	// TODO: add Debug mode
	// printf ( "\n#####################################\n\n\n\n\n\n" );
	// printf ( "  Input Data:\n" );
	// printf ( "\n" );

	// for (int i = 0; i < targetFrames; i++ )
	// {
	// 	printf ( "  %3d  %12f  %12f\n", i, targetSignal[i][0], targetSignal[i][1] );
	// }

	// fourrier transform of target and impulse signal
	target_plan_forward = fftw_plan_dft_1d(targetFrames, targetSignal, targetSignalFt, FFTW_FORWARD, FFTW_ESTIMATE);
	fftw_execute(target_plan_forward);

	impulse_plan_forward = fftw_plan_dft_1d(impulseFrames, impulseSignal, impulseSignalFt, FFTW_FORWARD, FFTW_ESTIMATE);
	fftw_execute(impulse_plan_forward);

	for (int i=0; i< targetFrames; i++){
		targetSignalFt[i][0]= ((impulseSignalFt[i][0]*targetSignalFt[i][0])-(impulseSignalFt[i][1]*targetSignalFt[i][1]));
		targetSignalFt[i][1]= ((impulseSignalFt[i][0]*targetSignalFt[i][1])+(impulseSignalFt[i][1]*targetSignalFt[i][0]));
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
	target_plan_backward = fftw_plan_dft_1d(targetFrames, targetSignalFt, targetSignalIft, FFTW_BACKWARD, FFTW_ESTIMATE);
	fftw_execute(target_plan_backward);

	// TODO: add Debug mode
	// printf ( "\n" );
	// printf ( "\n#####################################\n\n\n\n\n\n" );
	// printf ( "\n" );

	// for (int i = 0; i < targetFrames; i++ )
	// {
	// printf ( "  %3d  %12f  %12f\n", i, 
	// 	targetSignalIft[i][0] / ( double ) ( targetFrames ), targetSignalIft[i][1] / ( double ) ( targetFrames ) );
	// }

	// int diffs = 0;
	// for (int i = 0; i < targetFrames; ++i)
	// {
	// 	// TODO: add Debug mode
	// 	// if((float)((targetSignal[i][0])) != ((float)(targetSignalIft[i][0]) / (float) targetFrames)) {
	// 	// 	printf("diff at frame: %d\t (original) %f != (transformed) %f\n", i, (float)((targetSignal[i][0])), (float)((targetSignalIft[i][0])));
	// 	// 	diffs++;
	// 	// 	if (diffs >= 100) {
	// 	// 		break;
	// 	// 	}
	// 	// }
	// 	outputsx[i] = (float)((targetSignalIft[i][0]) / (float) targetFrames);
	// 	outputdx[i] = (float)((targetSignalIft[i][0]) / (float) targetFrames);
	// }

	float maxo[2];
	maxo[0]=0.0f;
	maxo[1]=0.0f;  

	for (int i = 0; i < targetFrames; i++){
		if (abs(maxo[0])<=abs(targetSignalIft[i][0])) maxo[0]=targetSignalIft[i][0];
		if (abs(maxo[1])<=abs(targetSignalIft[i][0])) maxo[1]=targetSignalIft[i][0];
	}
	float maxot= abs(maxo[0])>=abs(maxo[1])? abs(maxo[0]): abs(maxo[1]);

	for (int i=0; i< targetFrames; i++){
		float temp=0.0f;
		outputsx[i]= (float)((targetSignalIft[i][0])/(maxot));
		outputdx[i]= (float)((targetSignalIft[i][0])/(maxot));
	}

	delete [] targetSignal;
	delete [] targetSignalFt;
	delete [] targetSignalIft;

	return targetFrames;
}