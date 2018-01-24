

#include	<cstdio>
#include	<cstring>
#include	<cstdint>

#include	<sndfile.hh>

#include	"CPUconv.h"

#define		BUFFER_LEN		1024

static short impulseResponseBuffer [BUFFER_LEN];
static short targetSoundBuffer [BUFFER_LEN];


/*
 * create_file (fname, SF_FORMAT_WAV | SF_FORMAT_PCM_16);
 */
static void create_file (const char * fname, int format)
{	
	static short buffer [BUFFER_LEN] ;

	SndfileHandle file ;
	int channels = 2 ;
	int srate = 48000 ;

	printf ("Creating file named '%s'\n", fname) ;

	file = SndfileHandle (fname, SFM_WRITE, format, channels, srate) ;

	memset (buffer, 0, sizeof (buffer)) ;

	file.write (buffer, BUFFER_LEN) ;

	puts ("") ;
	/*
	**	The SndfileHandle object will automatically close the file and
	**	release all allocated memory when the object goes out of scope.
	**	This is the Resource Acquisition Is Initailization idom.
	**	See : http://en.wikipedia.org/wiki/Resource_Acquisition_Is_Initialization
	*/
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

	SndfileHandle targetSound = openTargetSound("resources/kick_for_breaks.wav");
	SndfileHandle impulseResponse = openImpulseResponse("resources/HallA.wav");

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

	targetSound.readf(targetSignal, targetSoundFrameCount * targetSoundChannelCount);
	impulseResponse.readf(impulseSignal, (impulseResponseChannelCount + 1) * impulseResponseChannelCount);

	float* impulsesx = new float[impulseResponseFrameCount];
	float* impulsedx = new float[impulseResponseFrameCount];

	// split up channels
	// for (int i = 0; i < impulseResponseFrameCount; i++) {
	// 	if(impulseResponseChannelCount==2){
	// 		impulsesx[i] = impulseSignal[2*i];
	// 		impulsedx[i] = impulseSignal[2*i+1];
	// 	}
	// 	else{
	// 		impulsesx[i] = impulseSignal[i];
	// 		impulsedx[i] = impulseSignal[i];
	// 	}
	// }

	// create output array for both channels
	float* outputsx = new float[targetSoundFrameCount + impulseResponseFrameCount - 1];
	float* outputdx = new float[targetSoundFrameCount + impulseResponseFrameCount - 1];

	CPUConv(targetSignal, impulsesx, impulsedx, outputsx, outputdx);

	return 0;
}