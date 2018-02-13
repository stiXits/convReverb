#include "GPUconvOAReverb.h"

#include <math.h>
#include <vector>
#include <array>
#include <clFFT.h>

//#ifndef ARM_COMPUTE_CL /* Needed by Utils.cpp to handle OpenCL exceptions properly */
////#error "This example needs to be built with -DARM_COMPUTE_CL"
//#endif /* ARM_COMPUTE_CL */
//
//#include "arm_compute/core/Types.h"
//#include "arm_compute/runtime/CL/CLFunctions.h"
//#include "arm_compute/runtime/CL/CLScheduler.h"
//#include "utils/Utils.h"
//
//using namespace arm_compute;

namespace gpuconv {
// FFT buffers
		float *paddedTargetSignalL;
    float *paddedTargetSignalR;
    float *impulseSignalL;
    float *impulseSignalR;

		std::vector<fftw_complex> mergedSignalL;
		std::vector<fftw_complex> mergedSignalR;

    cl_context ctx = 0;
    cl_command_queue queue = 0;

    void setUpCL(uint32_t bufferSize) {
//      CLScheduler::get().default_init();
      cl_int err;
      cl_platform_id platform = 0;
      cl_device_id device = 0;
      cl_context_properties props[3] = { CL_CONTEXT_PLATFORM, 0, 0 };

      cl_event event = NULL;
      int ret = 0;

      /* FFT library realted declarations */
      clfftPlanHandle planHandle;
      clfftDim dim = CLFFT_1D;

      /* Setup OpenCL environment. */
      err = clGetPlatformIDs( 1, &platform, NULL );
      err = clGetDeviceIDs( platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL );

      props[1] = (cl_context_properties)platform;
      ctx = clCreateContext( props, 1, &device, NULL, NULL, &err );
      queue = clCreateCommandQueue( ctx, device, 0, &err );

      /* Setup clFFT. */
      clfftSetupData fftSetup;
      err = clfftInitSetupData(&fftSetup);
      err = clfftSetup(&fftSetup);

      /* Allocate host & initialize data. */
      /* Only allocation shown for simplicity. */
//      X = (float *)malloc(bufferSize * 2 * sizeof(*X));

      /* Prepare OpenCL memory objects and place data inside them. */
//      targetBufferL = clCreateBuffer( ctx, CL_MEM_ALLOC_HOST_PTR, bufferSize * 2 * sizeof(float), NULL, &err );



      //   ------------->do cl fft here<---------------


//      return ret;
      printf("finished opencl setup with: %d\n", ret);
      return;
    }

    void tearDown()
    {
      /* Release clFFT library. */
      clfftTeardown( );

      /* Release OpenCL working objects. */
      clReleaseCommandQueue( queue );
      clReleaseContext( ctx );
    }

    void fft(float *buffer, uint32_t bufferSize, clfftDirection direction, cl_command_queue queue, cl_context ctx)
    {
      printf("begin fft\n");

      size_t clLengths[1] = {bufferSize};
      clfftPlanHandle planHandle;
      clfftDim dim = CLFFT_1D;

      /* Fetch results of calculations. */
      cl_int err;

      cl_mem bufferHandle = clCreateBuffer( ctx, CL_MEM_ALLOC_HOST_PTR , bufferSize * 2 * sizeof(float), NULL, &err );

      /* Create a default plan for a complex FFT. */
      err = clfftCreateDefaultPlan(&planHandle, ctx, dim, clLengths);
      printf("1: err: %d\n", err);

      /* Set plan parameters. */
      err = clfftSetPlanPrecision(planHandle, CLFFT_SINGLE);
      printf("2: err: %d\n", err);
      err = clfftSetLayout(planHandle, CLFFT_COMPLEX_INTERLEAVED, CLFFT_COMPLEX_INTERLEAVED);
      printf("3: err: %d\n", err);
      err = clfftSetResultLocation(planHandle, CLFFT_INPLACE);
      printf("4: err: %d\n", err);

      /* Bake the plan. */
      err = clfftBakePlan(planHandle, 1, &queue, NULL, NULL);
      printf("5: err: %d\n", err);
      /* Execute the plan. */
      err = clfftEnqueueTransform(planHandle, direction, 1, &queue, 0, NULL, NULL, &bufferHandle, NULL, NULL);
      printf("6: err: %d\n", err);

      /* Wait for calculations to be finished. */
      err = clFinish(queue);
      printf("7: err: %d\n", err);

      /* Fetch results of calculations. */
      err = clEnqueueReadBuffer( queue, bufferHandle, CL_TRUE, 0, bufferSize * 2 * sizeof(float), buffer, 0, NULL, NULL );
      printf("8: err: %d\n", err);

      err = clfftDestroyPlan( &planHandle );
      printf("9: err: %d\n", err);

      err = clReleaseMemObject(bufferHandle);
      printf("10: err: %d", err);

      printf("end fft: %d\n", err);
    }

    void padImpulseSignal(float *impulse, float *impulseBuffer, uint32_t  segmentSize)
    {
      uint32_t  transformedSegmentSize = 2 * segmentSize;

      // copy impulse sound to complex buffer
      for (int i = 0; i < transformedSegmentSize*2 ; i += 2) {
        if (i < segmentSize) {
          impulseBuffer[i] = impulse[i];
        } else {
          impulseBuffer[i] = 0.0f;
        }

        impulseBuffer[i + 1] = 0.0f;
      }
    }

		uint32_t
		oAReverb(float *target, uint32_t targetFrames, float *impulseL, float *impulseR, uint32_t impulseFrames,
										float *outputL, float *outputR) {

			fftw_plan impulseL_plan_forward, impulseR_plan_forward;
			uint32_t segmentCount = targetFrames / impulseFrames;
			uint32_t segmentSize = impulseFrames;
			uint32_t transformedSegmentSize = 2 * segmentSize;
			uint32_t transformedSignalSize = (transformedSegmentSize - 1) * segmentCount;

			impulseSignalL = new float[transformedSignalSize * 2];
			impulseSignalR = new float[transformedSignalSize * 2];
			paddedTargetSignalL = new float[transformedSignalSize * 2];
      paddedTargetSignalR = new float[transformedSignalSize * 2];

			// the resultsignal is impulsesize longer than the original
			mergedSignalL = std::vector<fftw_complex>(segmentSize * (segmentCount + 1));
			mergedSignalR = std::vector<fftw_complex>(segmentSize * (segmentCount + 1));

      setUpCL(transformedSegmentSize);

			padTargetSignal(target, segmentCount, segmentSize, paddedTargetSignalL);
      padTargetSignal(target, segmentCount, segmentSize, paddedTargetSignalR);
      padImpulseSignal(impulseL, impulseSignalL, impulseFrames);
      padImpulseSignal(impulseL, impulseSignalR, impulseFrames);

      fft(impulseSignalL, transformedSegmentSize, CLFFT_FORWARD, queue, ctx);
      fft(impulseSignalL, transformedSegmentSize, CLFFT_FORWARD, queue, ctx);

      transform(paddedTargetSignalL, impulseSignalL, transformedSegmentSize, segmentCount, queue, ctx);
      transform(paddedTargetSignalR, impulseSignalR, transformedSegmentSize, segmentCount, queue, ctx);

			float maxo[2];
			maxo[0] = 0.0f;
			maxo[1] = 0.0f;

//			maxo[0] = maximum(maxo[0], mergeConvolvedSignal(paddedTargetSignalL, mergedSignalL, segmentSize, segmentCount));
//			maxo[1] = maximum(maxo[1], mergeConvolvedSignal(paddedTargetSignalR, mergedSignalR, segmentSize, segmentCount));

			float maxot = abs(maxo[0]) >= abs(maxo[1]) ? abs(maxo[0]) : abs(maxo[1]);

			for (int i = 0; i < (targetFrames + impulseFrames - 1) * 2; i += 2) {
				outputL[i] = (float) ((paddedTargetSignalL[i]) / (maxot));
				outputR[i] = (float) ((paddedTargetSignalR[i]) / (maxot));
			}

      tearDown();

			return transformedSignalSize;
		}

		uint32_t transform(float *target,
											 float *impulse,
											 uint32_t sampleSize,
											 uint32_t segmentCount,
											 cl_command_queue queue,
											 cl_context) {

      printf("begin transform");
			cl_int err = 0;
			uint32_t transformedSegmentSize = 2 * sampleSize;

      printf("create buffer for target signal: %d\n", err);

			for (int i = 0; i < segmentCount; i += transformedSegmentSize * 2) {
				// conlvolve only parts of the input and output buffers
				convolve(&target[i], impulse, transformedSegmentSize, queue, ctx);
			}

      printf("end transform");
		}

		uint32_t convolve(float *target,
											float *impulse,
											uint32_t sampleSize,
                      cl_command_queue queue,
                      cl_context) {

      printf("begin convolve\n");

			// transform signal to frequency domaine
      fft(target, sampleSize, CLFFT_FORWARD, queue, ctx);

			// convolve target and signal
			for (int i = 0; i < sampleSize*2; i += 2) {

        target[i]     = ((impulse[i] * target[i]) - (impulse[i + 1] * target[i + 1]));
        target[i + 1] = ((impulse[i] * target[i + 1]) + (impulse[i + 1] * target[i]));
			}

      fft(target, sampleSize, CLFFT_BACKWARD, queue, ctx);

      printf("end convolve\n");

			return sampleSize;
		}

		uint32_t padTargetSignal(float *target, uint32_t segmentCount, uint32_t segmentSize,
														 float *destinationBuffer) {

			// cut the target signal into samplecount buffers
			uint32_t stride = segmentSize * 2 - 1;

			for (int i = 0; i < segmentCount*2; i += 2) {
				// copy targetsignal into new buffer
				for (int k = 0; k < segmentSize; ++k) {
					int offset = i * stride + k;
					destinationBuffer[offset] = target[offset];
					destinationBuffer[offset + 1] = 0.0f;
				}

				// pad the buffer with zeros til it reaches a size of samplesize * 2 - 1
				for (int k = 0; k < segmentSize*2; ++k) {
					int offset = i * stride + segmentSize + k;
					destinationBuffer[offset] = 0.0f;
					destinationBuffer[offset + 1] = 0.0f;
				}
			}

			return stride;
		}

		float mergeConvolvedSignal(std::vector<fftw_complex> &longInputBuffer, std::vector<fftw_complex> &shortOutpuBuffer,
															 uint32_t sampleSize, uint32_t sampleCount) {
			float max = 0;
			uint32_t stride = sampleSize * 2 - 1;
			// start with second sample, the first one has no signal tail to merge with
			for (int i = 0; i <= sampleCount; ++i) {
				uint32_t readHeadPosition = stride * i;
				// tail has length samplesize - 1 so the resulting + 1
				uint32_t readTailPosition = readHeadPosition - sampleSize + 1;
				uint32_t writePosition = sampleSize * i;

				for (int k = 0; k < sampleSize - 1; ++k) {
					if (i == 0) {
						// position is in an area where no tail exists, yet. Speaking the very first element:
						shortOutpuBuffer[writePosition + k][0] = longInputBuffer[readHeadPosition + k][0];
						shortOutpuBuffer[writePosition + k][1] = longInputBuffer[readHeadPosition + k][1];
						max = maximum(max, shortOutpuBuffer[writePosition + k][0]);
					}
//      else if (i == sampleCount) {
//        // segment add the last tail to output
//        shortOutpuBuffer[writePosition + k][0] = longInputBuffer[readTailPosition + k][0];
//        shortOutpuBuffer[writePosition + k][1] = longInputBuffer[readTailPosition + k][1];
//        max = maximum(max, shortOutpuBuffer[writePosition + k][0]);
//      } else {
//        // segment having a head and a tail to summ up
//        shortOutpuBuffer[writePosition + k][0] =
//                longInputBuffer[readHeadPosition][0] + longInputBuffer[readTailPosition][0];
//        shortOutpuBuffer[writePosition + k][1] =
//                longInputBuffer[readHeadPosition][1] + longInputBuffer[readTailPosition][1];
//        max = maximum(max, shortOutpuBuffer[writePosition + k][0]);
//      }
				}
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
			printf("\n#####################################\n\n\n\n\n\n");
			printf("Data (skipping zeros):\n");
			printf("\n");

			for (int i = 0; i < size; i++) {
				if (target[i][0] != 0.0f || target[i][1] != 0.0f)
					printf("  %3d  %12f  %12f\n", i, target[i][0], target[i][1]);
			}
		}

		void compareVectors(std::vector<fftw_complex> vec0, std::vector<fftw_complex> vec1, uint32_t size) {
			for (int i = 0; i < size; ++i) {
				if (vec0[0] != vec1[0] || vec0[1] != vec1[1]) {
					printf("Differing vectors:\n");
					printf("%12f  %12f\n", i, vec0[0], vec1[1]);
					printf("%12f  %12f\n", i, vec0[0], vec1[1]);
				}
			}
		}
}