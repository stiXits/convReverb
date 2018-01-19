

#include	<cstdio>
#include	<cstring>

#include	<sndfile.hh>

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
	SndfileHandle targetSound = openTargetSound("resources/kick_for_breaks.wav");
	SndfileHandle impulseResponse = openImpulseResponse("resources/HallA.wav");

	puts ("Done.\n") ;
	return 0;
}