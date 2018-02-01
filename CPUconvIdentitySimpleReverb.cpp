#include "CPUconvIdentitySimpleReverb.h"

#include <fftw3.h>
#include <math.h>

uint32_t CPUconvIdentitySimpleReverb(float *target, uint32_t targetFrames, float *impulsesx, float *impulsedx, uint32_t impulseFrames, float *outputsx, float *outputdx) {

	fftw_plan target_plan_forward, target_plan_backward;

	// FFT buffers
	fftw_complex* targetSignal = new fftw_complex[targetFrames];
	fftw_complex* targetSignalFt = new fftw_complex[targetFrames];
	fftw_complex* targetSignalIft = new fftw_complex[targetFrames];

	// prepare input signal for fft and move it to complex array
	for (int i = 0; i < targetFrames; ++i)
	{
		targetSignal[i][0] = target[i];
		targetSignal[i][1] = 0.0f;
	}

	// TODO: add Debug mode
	// printf ( "\n#####################################\n\n\n\n\n\n" );
	// printf ( "  Input Data:\n" );
	// printf ( "\n" );

	// for (int i = 0; i < targetFrames; i++ )
	// {
	// 	printf ( "  %3d  %12f  %12f\n", i, targetSignal[i][0], targetSignal[i][1] );
	// }

	// fourrier transform of targetSignal
	target_plan_forward = fftw_plan_dft_1d(targetFrames, targetSignal, targetSignalFt, FFTW_FORWARD, FFTW_ESTIMATE);
	fftw_execute(target_plan_forward);

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

	int diffs = 0;
	for (int i = 0; i < targetFrames; ++i)
	{
		// TODO: add Debug mode
		// if((float)((targetSignal[i][0])) != ((float)(targetSignalIft[i][0]) / (float) targetFrames)) {
		// 	printf("diff at frame: %d\t (original) %f != (transformed) %f\n", i, (float)((targetSignal[i][0])), (float)((targetSignalIft[i][0])));
		// 	diffs++;
		// 	if (diffs >= 100) {
		// 		break;
		// 	}
		// }
		outputsx[i] = (float)((targetSignalIft[i][0]) / (float) targetFrames);
		outputdx[i] = (float)((targetSignalIft[i][0]) / (float) targetFrames);
	}

	delete [] targetSignal;
	delete [] targetSignalFt;
	delete [] targetSignalIft;

	return targetFrames;
}