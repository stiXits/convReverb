#include "GPUconvOAReverb.h"

#include <math.h>

namespace gpuconv {

// FFT buffers
    std::vector<fftw_complex> impulseSignalL;
    std::vector<fftw_complex> impulseSignalLFT;

    std::vector<fftw_complex> impulseSignalR;
    std::vector<fftw_complex> impulseSignalRFT;

    std::vector<fftw_complex> paddedTargetSignal;

    std::vector<fftw_complex> intermediateSignalL;
    std::vector<fftw_complex> intermediateSignalR;

    std::vector<fftw_complex> convolvedSignalL;
    std::vector<fftw_complex> convolvedSignalR;

    std::vector<fftw_complex> mergedSignalL;
    std::vector<fftw_complex> mergedSignalR;

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

    cl_mem createGPUBuffer(std::vector<fftw_complex>::iterator &buffer, uint32_t bufferSize) {
      cl_int err = 0;
      cl_mem bufferHandle = clCreateBuffer( ctx, CL_MEM_ALLOC_HOST_PTR , bufferSize * 2 * sizeof(float), NULL, &err );
      err = clEnqueueWriteBuffer( queue, bufferHandle, CL_TRUE, 0, bufferSize * 2 * sizeof(float), &*buffer, 0, NULL, NULL );
      printf("enque write buffer: %d\n", err);

      return  bufferHandle;
    }

    void enqueueGPUWriteBuffer(cl_mem &bufferHandle, std::vector<fftw_complex>::iterator &buffer, uint32_t bufferSize) {
      cl_int err = 0;
      err = clEnqueueWriteBuffer( queue, bufferHandle, CL_TRUE, 0, bufferSize * 2 * sizeof(float), &*buffer, 0, NULL, NULL );
      printf("enque write buffer: %d\n", err);
    }

    void enqueueGPUReadBuffer(cl_mem &bufferHandle, std::vector<fftw_complex>::iterator &buffer, uint32_t bufferSize) {
      cl_int err = 0;
      err = clEnqueueReadBuffer( queue, bufferHandle, CL_TRUE, 0, bufferSize * 2 * sizeof(float), &*buffer, 0, NULL, NULL );
      printf("8: err: %d\n", err);
    }

    clfftPlanHandle createGPUPlan(uint32_t bufferSize) {
      clfftPlanHandle planHandle;
      cl_int err;
      clfftDim dim = CLFFT_1D;
      size_t clLengths[1] = {bufferSize};

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

      return planHandle;
    }

    void enqueueGPUPlan(clfftPlanHandle planHandle, cl_mem &bufferHandle, clfftDirection direction) {
      cl_int err = 0;
      err = clfftEnqueueTransform(planHandle, direction, 1, &queue, 0, NULL, NULL, &bufferHandle, NULL, NULL);
      printf("6: err: %d\n", err);
    }

    void fftParallel(std::vector<std::vector<fftw_complex>::iterator> buffers, uint32_t bufferSize, clfftDirection direction,
                     cl_command_queue queue, cl_context ctx) {
      printf("begin parallel fft\n");
      cl_int err = 0;

      std::vector<clfftPlanHandle> plans;
      std::vector<cl_mem> bufferHandles;

      // create all plans
      for(auto buffer: buffers) {
        bufferHandles.push_back(createGPUBuffer(buffer, bufferSize));
        plans.push_back(createGPUPlan(bufferSize));
      }

      // execute plans
      for(auto&& iteration: zip_range(plans, bufferHandles)) {
        enqueueGPUPlan(iteration.get<0>(), iteration.get<1>(), direction);
        err = clFinish(queue);
      }

      // read results
      for(auto&& iteration: zip_range(buffers, bufferHandles)) {
        enqueueGPUReadBuffer(iteration.get<1>(), iteration.get<0>(), bufferSize);
        err = clFinish(queue);
      }

      // clean up
      for(auto&& iteration: zip_range(plans, bufferHandles)) {
        clfftDestroyPlan(&iteration.get<0>());
        clReleaseMemObject(iteration.get<1>());
        err = clFinish(queue);
      }

    }

    void fftSingle(std::vector<fftw_complex>::iterator buffer, uint32_t bufferSize, clfftDirection direction,
                   cl_command_queue queue, cl_context ctx)
    {
      printf("begin single fft\n");

      /* Fetch results of calculations. */
      cl_int err = 0;

      cl_mem bufferHandle = createGPUBuffer(buffer, bufferSize);
      enqueueGPUWriteBuffer(bufferHandle, buffer, bufferSize);

      /* Create a default plan for a complex FFT. */
      clfftPlanHandle plan = createGPUPlan(bufferSize);

      enqueueGPUPlan(plan, bufferHandle, direction);

      /* Wait for calculations to be finished. */
      err = clFinish(queue);
      printf("7: err: %d\n", err);

      /* Fetch results of calculations. */
      enqueueGPUReadBuffer(bufferHandle, buffer, bufferSize);

      err = clfftDestroyPlan(&plan);
      printf("9: err: %d\n", err);

      err = clReleaseMemObject(bufferHandle);
      printf("10: err: %d\n", err);

      printf("end fft: %d\n", err);
    }

    uint32_t
    oAReverb(float *target, uint32_t targetFrames, float *impulseL, float *impulseR, uint32_t impulseFrames,
             float *outputL, float *outputR) {
      uint32_t segmentCount = targetFrames / impulseFrames;
      uint32_t segmentSize = impulseFrames;
      uint32_t transformedSegmentSize = 2 * segmentSize;
      uint32_t transformedSignalSize = (transformedSegmentSize) * segmentCount;

      impulseSignalL = std::vector<fftw_complex>(transformedSegmentSize);
      impulseSignalLFT = std::vector<fftw_complex>(transformedSegmentSize);

      impulseSignalR = std::vector<fftw_complex>(transformedSegmentSize);
      impulseSignalRFT = std::vector<fftw_complex>(transformedSegmentSize);

      paddedTargetSignal = std::vector<fftw_complex>(transformedSignalSize);

      intermediateSignalL = std::vector<fftw_complex>(transformedSignalSize);
      intermediateSignalR = std::vector<fftw_complex>(transformedSignalSize);

      convolvedSignalL = std::vector<fftw_complex>(transformedSignalSize);
      convolvedSignalR = std::vector<fftw_complex>(transformedSignalSize);

      // the resultsignal is impulsesize longer than the original
      mergedSignalL = std::vector<fftw_complex>(segmentSize * (segmentCount + 1));
      mergedSignalR = std::vector<fftw_complex>(segmentSize * (segmentCount + 1));

      setUpCL();

      padTargetSignal(target, segmentCount, segmentSize, transformedSegmentSize, paddedTargetSignal);

      padImpulseSignal(impulseL, impulseSignalL, segmentSize, transformedSegmentSize);
      padImpulseSignal(impulseR, impulseSignalR, segmentSize, transformedSegmentSize);

      fftSingle(impulseSignalL.begin(), transformedSegmentSize, CLFFT_FORWARD, queue, ctx);
      fftSingle(impulseSignalR.begin(), transformedSegmentSize, CLFFT_FORWARD, queue, ctx);

      std::vector<std::vector<fftw_complex>::iterator> buffersL;
      std::vector<std::vector<fftw_complex>::iterator> buffersR;
      // fourrier transform of target and impulse signal
      for (int i = 0; i < transformedSignalSize; i += transformedSegmentSize) {

        // todo: use buffers for left and right
        buffersL.push_back(paddedTargetSignal.begin() + i);
        buffersR.push_back(paddedTargetSignal.begin() + i);
        // chnlvolve only parts of the input and output buffers
//        convolve(paddedTargetSignal.begin() + i, impulseSignalL.begin(), transformedSegmentSize);
//        convolve(paddedTargetSignal.begin() + i, impulseSignalRFT.begin(), transformedSegmentSize);
      }
      convolveParallel(buffersL, impulseSignalL.begin(), transformedSegmentSize);
//      convolveParallel(buffersL, impulseSignalL.begin(), transformedSegmentSize);

      float maxo[2];
      maxo[0] = 0.0f;
      maxo[1] = 0.0f;

      maxo[0] = maximum(maxo[0], mergeConvolvedSignal(paddedTargetSignal, mergedSignalL, segmentSize, segmentCount));
      maxo[1] = maximum(maxo[1], mergeConvolvedSignal(paddedTargetSignal, mergedSignalR, segmentSize, segmentCount));

//      maxo[0] = maximum(maxo[0], mergeConvolvedSignal(impulseSignalL, mergedSignalL, segmentSize, segmentCount));
//      maxo[1] = maximum(maxo[1], mergeConvolvedSignal(impulseSignalR, mergedSignalR, segmentSize, segmentCount));

//      for (int j = 0; j < transformedSignalSize; j++) {
//        maxo[0] = maximum(maxo[0], impulseSignalL[j][0]);
//        maxo[1] = maximum(maxo[1], impulseSignalR[j][0]);
//      }
//
      float maxot = abs(maxo[0]) >= abs(maxo[1]) ? abs(maxo[0]) : abs(maxo[1]);
//
//      for (int i = 0; i < transformedSignalSize; i++) {
//        outputL[i] = (float) ((impulseSignalL[i][0]) / (maxot));
//        outputR[i] = (float) ((impulseSignalR[i][0]) / (maxot));
//      }

      for (int i = 0; i < targetFrames + impulseFrames - 1; i++) {
        outputL[i] = (float) ((mergedSignalL[i][0]) / (maxot));
        outputR[i] = (float) ((mergedSignalR[i][0]) / (maxot));
      }

      tearDown();

      return segmentSize * (segmentCount + 1);
    }

    uint32_t convolveParallel(std::vector<std::vector<fftw_complex>::iterator> targetSignals,
                              std::vector<fftw_complex>::iterator impulseSignal,
                              uint32_t bufferSize) {
      std::vector<std::vector<fftw_complex >::iterator> cachedTargetIterators = targetSignals;
      std::vector<fftw_complex >::iterator cachedImpulseIterator = impulseSignal;

      fftParallel(targetSignals, bufferSize, CLFFT_FORWARD, queue, ctx);

      // do complex multiplication
      for(auto buffer: targetSignals)
      {
        std::vector<fftw_complex >::iterator localImplulseSignal = cachedImpulseIterator;
        for (int i = 0; i < bufferSize; i++) {
          float cacheReal = ((*localImplulseSignal)[0] * (*buffer)[0] - (*localImplulseSignal)[1] * (*buffer)[1]);
          float cacheImaginary = ((*localImplulseSignal)[0] * (*buffer)[1] + (*localImplulseSignal)[1] * (*buffer)[0]);
          (*buffer)[0] = cacheReal;
          (*buffer)[1] = cacheImaginary;

          localImplulseSignal++;
          buffer++;
        }
      }

      fftParallel(targetSignals, bufferSize, CLFFT_BACKWARD, queue, ctx);

      return bufferSize;
    }

    uint32_t convolve(std::vector<fftw_complex>::iterator targetSignal,
                      std::vector<fftw_complex>::iterator impulseSignal,
                      uint32_t bufferSize) {

      std::vector<fftw_complex >::iterator cachedTargetSignalStart = targetSignal;
      // transform signal to frequency domaine
      fftSingle(targetSignal, bufferSize, CLFFT_FORWARD, queue, ctx);

      for (int i = 0; i < bufferSize; i++) {
        float cacheReal = ((*impulseSignal)[0] * (*targetSignal)[0] - (*impulseSignal)[1] * (*targetSignal)[1]);
        float cacheImaginary = ((*impulseSignal)[0] * (*targetSignal)[1] + (*impulseSignal)[1] * (*targetSignal)[0]);
        (*targetSignal)[0] = cacheReal;
        (*targetSignal)[1] = cacheImaginary;

        impulseSignal++;
        targetSignal++;
      }

      // transform result back to time domaine
      fftSingle(cachedTargetSignalStart, bufferSize, CLFFT_BACKWARD, queue, ctx);

      return bufferSize;
    }

    void padImpulseSignal(float *impulse, std::vector<fftw_complex> &impulseBuffer, uint32_t  segmentSize, uint32_t transformedSegmentSize)
    {
      // copy impulse sound to complex buffer
      for (int i = 0; i < transformedSegmentSize ; i++) {
        if (i < segmentSize) {
          impulseBuffer[i][0] = impulse[i];
        } else {
          impulseBuffer[i][0] = 0.0f;
        }

        impulseBuffer[i][1] = 0.0f;
      }
    }

    void padTargetSignal(float *target, uint32_t segmentCount, uint32_t segmentSize, uint32_t transformedSegmentSize,
                             std::vector<fftw_complex> &destinationBuffer) {

      for (int i = 0; i < segmentCount; ++i) {
        // copy targetsignal into new buffer
        for (int k = 0; k < segmentSize; ++k) {
          int readOffset = segmentSize * i + k;
          int writeOffset = segmentSize * 2 * i + k;

          destinationBuffer[writeOffset][0] = target[readOffset];
          destinationBuffer[writeOffset][1] = 0.0f;
        }

        // pad the buffer with zeros til it reaches a size of samplesize * 2 - 1
        for (int k = 0; k < transformedSegmentSize - segmentSize; ++k) {
          int writeOffset = segmentSize * i * 2 + segmentSize +  k;
          destinationBuffer[writeOffset][0] = 0.0f;
          destinationBuffer[writeOffset][1] = 0.0f;
        }
      }
    }

    float mergeConvolvedSignal(std::vector<fftw_complex> &longInputBuffer, std::vector<fftw_complex> &shortOutputBuffer,
                               uint32_t sampleSize, uint32_t sampleCount) {
      float max = 0;
      uint32_t stride = sampleSize * 2;
      // start with second sample, the first one has no signal tail to merge with
      for (int i = 0; i <= sampleCount; ++i) {
        uint32_t readHeadPosition = stride * i;
        // tail has length samplesize - 1 so the resulting + 1
        uint32_t readTailPosition = readHeadPosition - sampleSize;
        uint32_t writePosition = sampleSize * i;

        for (int k = 0; k < sampleSize; k++) {
          if (i == 0) {
            // position is in an area where no tail exists, yet. Speaking the very first element:
            shortOutputBuffer[writePosition + k][0] = (float)(longInputBuffer[readHeadPosition + k][0]);
            shortOutputBuffer[writePosition + k][1] = (float)(longInputBuffer[readHeadPosition + k][1]);
            max = maximum(max, shortOutputBuffer[writePosition + k][0]);
          }
          else if (i == sampleCount) {
            // segment add the last tail to output
            shortOutputBuffer[writePosition + k][0] = (float)(longInputBuffer[readTailPosition + k][0]);
            shortOutputBuffer[writePosition + k][1] = (float)(longInputBuffer[readTailPosition + k][1]);
            max = maximum(max, shortOutputBuffer[writePosition + k][0]);
          } else {
            // segment having a head and a tail to summ up
            shortOutputBuffer[writePosition + k][0] =
                    (float)(longInputBuffer[readHeadPosition + k][0] + longInputBuffer[readTailPosition + k][0]);
            shortOutputBuffer[writePosition + k + 1][0] =
                    (float)(longInputBuffer[readHeadPosition + k][1] + longInputBuffer[readTailPosition + k][1]);
            max = maximum(max, shortOutputBuffer[writePosition + k][0]);
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