#include	<cstdio>
#include	<cstring>
#include	<cstdint>

#include	<sndfile.hh>

#include	"CPUconv.h"

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
	sf_count_t sampleread;

	SndfileHandle targetSound = openTargetSound("resources/crash18.wav");
	SndfileHandle impulseResponse = openImpulseResponse("resources/HallA.wav");
	SndfileHandle outputSound = create_file("resources/output.wav", SF_FORMAT_WAV | SF_FORMAT_PCM_16);

	uint32_t outputSize = 0;
	uint32_t targetSoundFrameCount = (uint32_t) targetSound.frames();
	uint8_t targetSoundChannelCount = (uint8_t) targetSound.channels();
	uint32_t impulseResponseFrameCount = (uint32_t) impulseResponse.frames();
	uint8_t impulseResponseChannelCount = (uint8_t) impulseResponse.channels();

	printf("target sound sample count %d\n", targetSoundFrameCount);
	printf("target sound channel count %d\n", targetSoundChannelCount);
	printf("impuls response sample count %d\n", impulseResponseFrameCount);
	printf("impuls response channel count %d\n", impulseResponseChannelCount);

	float* targetSignal = new float[targetSoundFrameCount * targetSoundChannelCount];
	float* impulseSignal = new float[impulseResponseFrameCount * impulseResponseChannelCount];
	float* outputSoundSignal;

	targetSound.read(targetSignal, targetSoundFrameCount * targetSoundChannelCount);
	impulseResponse.read(impulseSignal, impulseResponseChannelCount * impulseResponseChannelCount);

	float* impulsesx = new float[impulseResponseFrameCount];
	float* impulsedx = new float[impulseResponseFrameCount];

	//split up channels
	for (int i = 0; i < impulseResponseFrameCount; i++) {
		if(impulseResponseChannelCount==2){
			impulsesx[i] = impulseSignal[2*i];
			impulsedx[i] = impulseSignal[2*i+1];
		}
		else{
			impulsesx[i] = impulseSignal[i];
			impulsedx[i] = impulseSignal[i];
		}
	}

	// create output array for both channels
	float* outputsx = new float[targetSoundFrameCount + impulseResponseFrameCount - 1];
	float* outputdx = new float[targetSoundFrameCount + impulseResponseFrameCount - 1];

	outputSize = CPUConv(targetSignal, targetSoundFrameCount, impulsesx, impulsedx, impulseResponseFrameCount, outputsx, outputdx);

	printf("outputSize: %d\n", outputSize);

	// write results to outputfile
	outputSoundSignal = new float[outputSize * targetSoundChannelCount];
	for (int i = 0; i < outputSize; i++) {
		outputSoundSignal[2*i]=(outputsx[i]);
		outputSoundSignal[2*i+1]=(outputdx[i]);
	}

	outputSound.write(outputSoundSignal, outputSize * targetSoundChannelCount) ;

	return 0;
}