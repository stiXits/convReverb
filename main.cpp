#include	<cstdio>
#include	<cstdint>

#include	<sndfile.hh>

#include  "GPUconvOAReverb.h"
#include "CPUconvSimpleReverb.h"
#include "CPUconvOAReverb.h"

#define		BUFFER_LEN		1024

static short impulseResponseBuffer[BUFFER_LEN];
static short targetSoundBuffer[BUFFER_LEN];

/*
 * create_file (fname, SF_FORMAT_WAV | SF_FORMAT_PCM_16);
 */
static SndfileHandle create_file (const char * fname, int format)
{	

	SndfileHandle file ;

	int channels = 2 ;
	int srate = 48000 ;

	printf ("Creating file named '%s'\n", "resources/output.wav") ;

	file = SndfileHandle(fname, SFM_WRITE, format, channels, srate) ;


	return file;
} /* create_file */

static SndfileHandle openImpulseResponse(const char * fname) {
	SndfileHandle file;
	file = SndfileHandle(fname) ;
	file.read(impulseResponseBuffer, BUFFER_LEN);
	return file;
}

static SndfileHandle openTargetSound(const char * fname) {
	SndfileHandle file;
	file = SndfileHandle(fname);
	file.read(targetSoundBuffer, BUFFER_LEN);
	return file;
}

int main(int argc, char const *argv[]) {
	SndfileHandle targetSound = openTargetSound("resources/basic_kick2.wav");
	SndfileHandle impulseResponse = openImpulseResponse("resources/HallA_X.wav");
	SndfileHandle outputSound = create_file("resources/output.wav", SF_FORMAT_WAV | SF_FORMAT_PCM_16);

	uint32_t outputSize = 0;
	uint32_t targetSoundFrameCount = (uint32_t) targetSound.frames();
	int targetSoundChannelCount = targetSound.channels();
	uint32_t impulseResponseFrameCount = (uint32_t) impulseResponse.frames();
	int impulseResponseChannelCount = impulseResponse.channels();

	printf("target sound sample count %d\n", targetSoundFrameCount);
	printf("target sound channel count %d\n", targetSoundChannelCount);
	printf("impuls response sample count %d\n", impulseResponseFrameCount);
	printf("impuls response channel count %d\n", impulseResponseChannelCount);

	float* targetSignal = new float[targetSoundFrameCount * targetSoundChannelCount];
	float* impulseSignal = new float[impulseResponseFrameCount * impulseResponseChannelCount];
	float* outputSoundSignal;

	SNDFILE      *infile1, *infile2;
	SF_INFO      sfinfo1, sfinfo2 ;
  uint32_t          samp1, samp2;
	uint32_t          sampleread;
	int chan1, chan2;

	targetSound.readf(targetSignal, targetSoundFrameCount * targetSoundChannelCount);
	impulseResponse.readf(impulseSignal, impulseResponseChannelCount * impulseResponseChannelCount);
    
    
	infile1 = sf_open ("resources/basic_kick2.wav", SFM_READ, &sfinfo1);
	if (infile1 == NULL)
	{   
		printf ("Unable to open file 1.\n");
		return  1 ;
	}
	infile2 = sf_open ("resources/HallA_X.wav", SFM_READ, &sfinfo2);
	if (infile2 == NULL)
	{  
		printf ("Unable to open file 2.\n");
		return  1 ;
	}
    
    
	samp1=sfinfo1.samplerate;
	samp2=sfinfo2.samplerate;
	chan1=sfinfo1.channels;
	chan2=sfinfo2.channels;
    
	if (samp1!=samp2){
		printf("Error, Sample Rate mismatch.\n");
		return 1;
	}

	if (((targetSoundChannelCount==2)&&(impulseResponseChannelCount==2))||((targetSoundChannelCount>1)||(impulseResponseChannelCount>2))){
		printf("Unable to perform convolution with two stereo or multichannel files.\n");
		return 1;
	}




    
	targetSignal= (float*)malloc(sizeof(float) * targetSoundFrameCount*targetSoundChannelCount);
	// Allocate host memory for the signal
	sampleread = sf_read_float (infile1, targetSignal, targetSoundFrameCount*chan1);
	if (sampleread != targetSoundFrameCount ) 
	{ 
		printf ("Error reading target sound");
		return 1;
	}
    
	impulseSignal= (float*)malloc(sizeof(float) * impulseResponseFrameCount*chan2);
	// Allocate host memory for the filter  
	sampleread = sf_read_float (infile2, impulseSignal, impulseResponseFrameCount*chan2);
	if (sampleread != impulseResponseFrameCount * impulseResponseChannelCount )
	{ 
		printf ("Error reading impulse response, %d", sampleread);
		return 1;
	}
	// ###############################################
    
    
	float* filtersx= (float*)malloc(sizeof(float) * impulseResponseFrameCount);
	float* filterdx= (float*)malloc(sizeof(float) * impulseResponseFrameCount);
    
	for (int i = 0; i < impulseResponseFrameCount; i++) {
		if(impulseResponseChannelCount==2){
			filtersx[i] = impulseSignal[2*i];
			filterdx[i] = impulseSignal[2*i+1];
		}
		else{
			filtersx[i] = impulseSignal[i];
			filterdx[i] = impulseSignal[i];
		}
	}

	// create output array for both channels
  uint32_t segmentCount = targetSoundFrameCount / impulseResponseFrameCount;
  uint32_t segmentSize = impulseResponseFrameCount;
  uint32_t transformedSegmentSize = 2 * segmentSize - 1;
  uint32_t transformedSignalSize = (transformedSegmentSize) * segmentCount;
	float* outputsx=(float*)malloc(sizeof(float) * (transformedSignalSize));
	float* outputdx=(float*)malloc(sizeof(float) * (transformedSignalSize));

//	outputSize = cpuconv::oAReverb(targetSignal, 65536*2, filtersx, filterdx, 4096*2, outputsx, outputdx);
//  outputSize = cpuconv::oAReverb(targetSignal, 512, filtersx, filterdx, 128 , outputsx, outputdx);
  outputSize = cpuconv::oAReverb(targetSignal, targetSoundFrameCount, filtersx, filterdx, impulseResponseFrameCount, outputsx, outputdx);
//	outputSize = CPUconv(targetSignal, targetSoundFrameCount, filtersx, filterdx, impulseResponseFrameCount, outputsx, outputdx,1);

	uint32_t outputLength = outputSize * 2  + 1;
	printf("outputSize: \t\t%d\n", outputSize);

	printf("outpusx: \t\t%d\n", transformedSignalSize);
	printf("outpusx: \t\t%d\n", transformedSignalSize);
	printf("outputSoundSignal: \t%d\n", outputLength);
	printf("returned size: \t\t%d\n", outputSize);

	// write results to outputfile
	outputSoundSignal = new float[outputLength];
	for (int i = 0; i < outputSize; i++) {
		outputSoundSignal[2*i]=(outputsx[i]);
		outputSoundSignal[2*i+1]=(outputdx[i]);
	}

	outputSound.write(outputSoundSignal, outputSize * 2 ) ;

	return 0;
}