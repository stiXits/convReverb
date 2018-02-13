#include "GPUconvOAReverb.h"

#include <math.h>

namespace gpuconv {
// FFT buffers
		float *paddedTargetSignalL;
    float *paddedTargetSignalR;
    float *impulseSignalL;
    float *impulseSignalR;

		float *mergedSignalL;
		float *mergedSignalR;

    cl_context ctx = 0;
    cl_command_queue queue = 0;

    void setUpCL() {
      cl_int err;
      cl_platform_id platform = 0;
      cl_device_id device = 0;
      cl_context_properties props[3] = { CL_CONTEXT_PLATFORM, 0, 0 };

      int ret = 0;

      /* Setup OpenCL environment. */
      err += clGetPlatformIDs( 1, &platform, NULL );
      err += clGetDeviceIDs( platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL );

      props[1] = (cl_context_properties)platform;
      ctx = clCreateContext( props, 1, &device, NULL, NULL, &err );
      queue = clCreateCommandQueue( ctx, device, 0, &err );

      /* Setup clFFT. */
      clfftSetupData fftSetup;
      err += clfftInitSetupData(&fftSetup);
      err += clfftSetup(&fftSetup);
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
      cl_int err = 0;

      cl_mem bufferHandle = clCreateBuffer( ctx, CL_MEM_ALLOC_HOST_PTR , bufferSize * 2 * sizeof(float), NULL, &err );
      err = clEnqueueWriteBuffer( queue, bufferHandle, CL_TRUE, 0, bufferSize * 2 * sizeof(float), buffer, 0, NULL, NULL );
      printf("enque write buffer: %d\n", err);

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

			uint32_t segmentCount = targetFrames / impulseFrames;
			uint32_t segmentSize = impulseFrames;
			uint32_t transformedSegmentSize = 2 * segmentSize;
			uint32_t transformedSignalSize = (transformedSegmentSize) * segmentCount;

      printf("segmentcount: %d\n segmentSize: %d\n transformedSegmentSize: %d\n transformedSignalSize: %d\n",
             segmentCount, segmentSize, transformedSegmentSize, transformedSignalSize);

			impulseSignalL = new float[transformedSignalSize * 2];
			impulseSignalR = new float[transformedSignalSize * 2];
      paddedTargetSignalR = new float[transformedSignalSize * 2];
      paddedTargetSignalL = new float[transformedSignalSize * 2];

			// the resultsignal is impulsesize longer than the original
			mergedSignalL = new float[transformedSignalSize * 2];
			mergedSignalR = new float[transformedSignalSize * 2];

      setUpCL();

			padTargetSignal(target, segmentCount, segmentSize, paddedTargetSignalL);
      padTargetSignal(target, segmentCount, segmentSize, paddedTargetSignalR);
      padImpulseSignal(impulseL, impulseSignalL, impulseFrames);
      padImpulseSignal(impulseL, impulseSignalR, impulseFrames);

      fft(impulseSignalL, transformedSegmentSize, CLFFT_FORWARD, queue, ctx);
      fft(impulseSignalR, transformedSegmentSize, CLFFT_FORWARD, queue, ctx);

      transform(paddedTargetSignalL, impulseSignalL, transformedSegmentSize, segmentCount, queue, ctx);
      transform(paddedTargetSignalR, impulseSignalR, transformedSegmentSize, segmentCount, queue, ctx);

			float maxo[2];
			maxo[0] = 0.0f;
			maxo[1] = 0.0f;

			maxo[0] = maximum(maxo[0], mergeConvolvedSignal(paddedTargetSignalL, mergedSignalL, segmentSize, segmentCount));
			maxo[1] = maximum(maxo[1], mergeConvolvedSignal(paddedTargetSignalR, mergedSignalR, segmentSize, segmentCount));
//
//      for (int j = 0; j < transformedSignalSize * 2; j += 2) {
//        maxo[0] = maximum(maxo[0], paddedTargetSignalL[j]);
//        maxo[1] = maximum(maxo[1], paddedTargetSignalR[j]);
//      }

			float maxot = abs(maxo[0]) >= abs(maxo[1]) ? abs(maxo[0]) : abs(maxo[1]);

			for (int i = 0; i < transformedSignalSize; i++) {
        outputL[i] = ((mergedSignalL[i * 2]) / (maxot));
        outputR[i] = ((mergedSignalR[i * 2]) / (maxot));
//				outputL[i] = ((paddedTargetSignalL[i * 2]) / (maxot));
//				outputR[i] = ((paddedTargetSignalR[i * 2]) / (maxot));
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
      printf("begin transform\n");
			cl_int err = 0;
      printf("create buffer for target signal: %d\n", err);
			for (int i = 0; i < segmentCount; i++) {
				// conlvolve only parts of the input and output buffers
				convolve(&target[i * sampleSize * 2], impulse, sampleSize, queue, ctx);
			}
      printf("end transform\n");
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
			for (int i = 0; i < segmentCount; i++) {
				// copy targetsignal into new buffer
				for (int k = 0; k < segmentSize*2; k += 2) {
					int readOffset = segmentSize * 2 * i + k;
          int writeOffset = segmentSize * 4 * i + k;
					destinationBuffer[writeOffset] = target[readOffset];
					destinationBuffer[writeOffset + 1] = 0.0f;
//          printf("%d = %f\n", writeOffset, target[readOffset]);
//          printf("%d = %f\n", writeOffset + 1, 0.0f);
        }

				// pad the buffer with zeros til it reaches a size of samplesize * 2 - 1
				for (int k = 0; k < segmentSize*2; k += 2) {
					int writeOffset = segmentSize * i * 4 + 2 * segmentSize +  k;
					destinationBuffer[writeOffset] = 0.0f;
					destinationBuffer[writeOffset + 1] = 0.0f;
//          printf("%d = %f\n", writeOffset, 0.0f);
//          printf("%d = %f\n", writeOffset + 1, 0.0f);
				}
			}

		}

		float mergeConvolvedSignal(float *longInputBuffer, float *shortOutpuBuffer,
															 uint32_t sampleSize, uint32_t sampleCount) {

      printArray(longInputBuffer, sampleSize * 4 * sampleCount);
			float max = 0;
			uint32_t stride = sampleSize * 4;
			// start with second sample, the first one has no signal tail to merge with
			for (int i = 0; i <= sampleCount; ++i) {
				uint32_t readHeadPosition = stride * i;
				// tail has length samplesize - 1 so the resulting + 1
				uint32_t readTailPosition = readHeadPosition - 2 * sampleSize;
				uint32_t writePosition = 2 * sampleSize * i;

				for (int k = 0; k < sampleSize * 2; k += 2) {
					if (i == 0) {
						// position is in an area where no tail exists, yet. Speaking the very first element:
						shortOutpuBuffer[writePosition + k] = longInputBuffer[readHeadPosition + k];
						shortOutpuBuffer[writePosition + k + 1] = longInputBuffer[readHeadPosition + k + 1];
						max = maximum(max, shortOutpuBuffer[writePosition + k]);
					}
        else if (i == sampleCount) {
          // segment add the last tail to output
          shortOutpuBuffer[writePosition + k] = longInputBuffer[readTailPosition + k];
          shortOutpuBuffer[writePosition + k + 1] = longInputBuffer[readTailPosition + k + 1];
          max = maximum(max, shortOutpuBuffer[writePosition + k]);
        } else {
          // segment having a head and a tail to summ up
          shortOutpuBuffer[writePosition + k] =
                  longInputBuffer[readHeadPosition + k] + longInputBuffer[readTailPosition + k];
          shortOutpuBuffer[writePosition + k + 1] =
                  longInputBuffer[readHeadPosition + k + 1] + longInputBuffer[readTailPosition + k + 1];
          max = maximum(max, shortOutpuBuffer[writePosition + k]);
        }
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

		void printArray(float *target, uint32_t size) {
			printf("\n#####################################\n\n\n\n\n\n");
			printf("Data (skipping zeros):\n");
			printf("\n");

			for (int i = 0; i < size; i++) {
        bool condition = (abs(target[i]) > 0.01);
//				if(condition) {
          printf("  %3d  %12f \n", i, target[i]);
//        }
			}
		}

		void compareVectors(float *vec0, float *vec1, uint32_t size) {
      printf("Differing vectors:\n");
			for (int i = 0; i < size; ++i) {
				if (vec0[0] != vec1[0] || vec0[1] != vec1[1]) {
					printf("%12f  %12f\n", i, vec0[0], vec1[1]);
					printf("%12f  %12f\n", i, vec0[0], vec1[1]);
				}
			}
		}
}