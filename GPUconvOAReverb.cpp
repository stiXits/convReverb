#include "GPUconvOAReverb.h"

#include <math.h>
#include <ctime>

namespace gpuconv {

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
      // printf("finished opencl setup with: %d\n", ret);
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

    cl_mem createGPUBuffer(std::vector<complex>::iterator &buffer, uint32_t bufferSize) {
      cl_int err = 0;
      cl_mem bufferHandle = clCreateBuffer( ctx, CL_MEM_ALLOC_HOST_PTR , bufferSize * 2 * sizeof(float), NULL, &err );
      // printf("enque write buffer: %d\n", err);

      return  bufferHandle;
    }

    void enqueueGPUWriteBuffer(cl_mem &bufferHandle, std::vector<complex>::iterator &buffer, uint32_t bufferSize) {
      cl_int err = 0;
      err = clEnqueueWriteBuffer( queue, bufferHandle, CL_TRUE, 0, bufferSize * 2 * sizeof(float), &*buffer, 0, NULL, NULL );
      // printf("enque write buffer: %d\n", err);
    }

    void enqueueGPUReadBuffer(cl_mem &bufferHandle, std::vector<complex>::iterator &buffer, uint32_t bufferSize) {
      cl_int err = 0;
      err = clEnqueueReadBuffer( queue, bufferHandle, CL_TRUE, 0, bufferSize * 2 * sizeof(float), &*buffer, 0, NULL, NULL );
      // printf("8: err: %d\n", err);
    }

    clfftPlanHandle createGPUPlan(uint32_t bufferSize) {
      clfftPlanHandle planHandle;
      cl_int err;
      clfftDim dim = CLFFT_1D;
      size_t clLengths[1] = {bufferSize};

      err = clfftCreateDefaultPlan(&planHandle, ctx, dim, clLengths);
      err = clfftSetPlanPrecision(planHandle, CLFFT_SINGLE);
      err = clfftSetLayout(planHandle, CLFFT_COMPLEX_INTERLEAVED, CLFFT_COMPLEX_INTERLEAVED);
      err = clfftSetResultLocation(planHandle, CLFFT_INPLACE);

      // bake the plan
      err = clfftBakePlan(planHandle, 1, &queue, NULL, NULL);

      return planHandle;
    }

    void enqueueGPUPlan(clfftPlanHandle planHandle, cl_mem &bufferHandle, clfftDirection direction) {
      cl_int err = 0;
      err = clfftEnqueueTransform(planHandle, direction, 1, &queue, 0, NULL, NULL, &bufferHandle, NULL, NULL);
    }

    void fftParallel(std::vector<std::vector<complex>::iterator> buffers, uint32_t bufferSize, clfftDirection direction,
                     cl_command_queue queue) {
      // printf("begin parallel fft\n");
      cl_int err = 0;

      std::vector<clfftPlanHandle> plans;
      std::vector<cl_mem> bufferHandles;

      // create all plans
      for(auto buffer: buffers) {
        cl_mem bufferHandle = createGPUBuffer(buffer, bufferSize);
        enqueueGPUWriteBuffer(bufferHandle, buffer, bufferSize);
        bufferHandles.push_back(bufferHandle);
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

    void fftSingle(std::vector<complex>::iterator buffer, uint32_t bufferSize, clfftDirection direction,
                   cl_command_queue queue)
    {
      // printf("begin single fft\n");
      cl_int err = 0;

      cl_mem bufferHandle = createGPUBuffer(buffer, bufferSize);
      enqueueGPUWriteBuffer(bufferHandle, buffer, bufferSize);
      clfftPlanHandle plan = createGPUPlan(bufferSize);
      enqueueGPUPlan(plan, bufferHandle, direction);
      err = clFinish(queue);
      enqueueGPUReadBuffer(bufferHandle, buffer, bufferSize);
      err = clReleaseMemObject(bufferHandle);
      err = clfftDestroyPlan(&plan);
    }

    uint32_t
    oAReverb(float* &target, uint32_t targetFrames, float* &impulseL, float* &impulseR, uint32_t impulseFrames,
             float* &outputL, float* &outputR) {
      uint32_t segmentCount = targetFrames / impulseFrames;
      if(segmentCount < 1) {
        segmentCount = 1;
      }
      uint32_t segmentSize = impulseFrames;
      uint32_t transformedSegmentSize = 2 * segmentSize;
      uint32_t transformedSignalSize = (transformedSegmentSize) * segmentCount;

      std::vector<complex> impulseSignalL(transformedSegmentSize);
      std::vector<complex> impulseSignalR(transformedSegmentSize);

      std::vector<complex> paddedTargetSignalL(transformedSignalSize);
      std::vector<complex> paddedTargetSignalR(transformedSignalSize);

      // the resultsignal is impulsesize longer than the original
      std::vector<complex> mergedSignalL(segmentSize * (segmentCount + 1));
      std::vector<complex> mergedSignalR(segmentSize * (segmentCount + 1));

      setUpCL();

      padTargetSignal(target, segmentCount, segmentSize, transformedSegmentSize, paddedTargetSignalL);
      padTargetSignal(target, segmentCount, segmentSize, transformedSegmentSize, paddedTargetSignalR);

      padImpulseSignal(impulseL, impulseSignalL, segmentSize, transformedSegmentSize);
      padImpulseSignal(impulseR, impulseSignalR, segmentSize, transformedSegmentSize);

      fftSingle(impulseSignalL.begin(), transformedSegmentSize, CLFFT_FORWARD, queue);
      fftSingle(impulseSignalR.begin(), transformedSegmentSize, CLFFT_FORWARD, queue);

      std::vector<std::vector<complex>::iterator> buffersL;
      std::vector<std::vector<complex>::iterator> buffersR;

      clock_t begin = clock();
      // fourrier transform of target and impulse signal
      for (int i = 0; i < transformedSignalSize; i += transformedSegmentSize) {
        buffersL.push_back(paddedTargetSignalL.begin() + i);
        buffersR.push_back(paddedTargetSignalR.begin() + i);
      }

      convolveParallel(buffersL, impulseSignalL.begin(), transformedSegmentSize);
      convolveParallel(buffersR, impulseSignalR.begin(), transformedSegmentSize);
      clock_t end = clock();
      double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
      printf("time consumed for raw convolution: %f seconds\n", elapsed_secs);


      float maxo[2];
      maxo[0] = 0.0f;
      maxo[1] = 0.0f;

      maxo[0] = maximum(maxo[0], mergeConvolvedSignal(paddedTargetSignalL, mergedSignalL, segmentSize, segmentCount));
      maxo[1] = maximum(maxo[1], mergeConvolvedSignal(paddedTargetSignalR, mergedSignalR, segmentSize, segmentCount));

      float maxot = abs(maxo[0]) >= abs(maxo[1]) ? abs(maxo[0]) : abs(maxo[1]);

      for (int i = 0; i < targetFrames + impulseFrames - 1; i++) {
        outputL[i] = (mergedSignalL[i].real()) / (maxot);
        outputR[i] = (mergedSignalR[i].real()) / (maxot);
      }

      tearDown();

      return segmentSize * (segmentCount + 1);
    }

    uint32_t convolveParallel(std::vector<std::vector<complex>::iterator> targetSignals,
                              std::vector<complex>::iterator impulseSignal,
                              uint32_t bufferSize) {
      std::vector<std::vector<complex >::iterator> cachedTargetIterators = targetSignals;
      std::vector<complex >::iterator cachedImpulseIterator = impulseSignal;

      fftParallel(targetSignals, bufferSize, CLFFT_FORWARD, queue);

      // do complex multiplication
      for(auto buffer: targetSignals)
      {
        std::vector<complex >::iterator localImpulseSignal = cachedImpulseIterator;
        for (int i = 0; i < bufferSize; i++) {
          *buffer = (*localImpulseSignal) * (*buffer);
          localImpulseSignal++;
          buffer++;
        }
      }

      fftParallel(targetSignals, bufferSize, CLFFT_BACKWARD, queue);

      return bufferSize;
    }

    uint32_t convolve(std::vector<complex>::iterator targetSignal,
                      std::vector<complex>::iterator impulseSignal,
                      uint32_t bufferSize) {

      std::vector<complex >::iterator cachedTargetSignalStart = targetSignal;
      // transform signal to frequency domaine
      fftSingle(targetSignal, bufferSize, CLFFT_FORWARD, queue);

      for (int i = 0; i < bufferSize; i++) {
        *targetSignal = (*impulseSignal) * (*targetSignal);
        impulseSignal++;
        targetSignal++;
      }

      // transform result back to time domaine
      fftSingle(cachedTargetSignalStart, bufferSize, CLFFT_BACKWARD, queue);

      return bufferSize;
    }

    void padImpulseSignal(float *impulse, std::vector<complex> &impulseBuffer, uint32_t  segmentSize,
                          uint32_t transformedSegmentSize)
    {
      // copy impulse sound to complex buffer
      for (int i = 0; i < transformedSegmentSize ; i++) {
        if (i < segmentSize) {
          impulseBuffer[i] = complex(impulse[i], 0.f);
        } else {
          impulseBuffer[i] = complex(0.f, 0.f);
        }
      }
    }

    void padTargetSignal(float *target, uint32_t segmentCount, uint32_t segmentSize, uint32_t transformedSegmentSize,
                             std::vector<complex> &destinationBuffer) {

      for (int i = 0; i < segmentCount; ++i) {
        // copy targetsignal into new buffer
        for (int k = 0; k < segmentSize; ++k) {
          int readOffset = segmentSize * i + k;
          int writeOffset = transformedSegmentSize * i + k;
          destinationBuffer[writeOffset] = complex(target[readOffset], 0.f);
        }

        // pad the buffer with zeros til it reaches a size of samplesize * 2 - 1
        for (int k = 0; k < transformedSegmentSize - segmentSize; ++k) {
          int writeOffset = transformedSegmentSize + segmentSize +  k;
          destinationBuffer[writeOffset] = complex(0.f, 0.f);
        }
      }
    }

    float mergeConvolvedSignal(std::vector<complex> &longInputBuffer, std::vector<complex> &shortOutputBuffer,
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
            shortOutputBuffer[writePosition + k] = longInputBuffer[readHeadPosition + k];
            max = maximum(max, shortOutputBuffer[writePosition + k].real());
          }
          else if (i == sampleCount) {
            // segment add the last tail to output
            shortOutputBuffer[writePosition + k] = longInputBuffer[readTailPosition + k];
            max = maximum(max, shortOutputBuffer[writePosition + k].real());
          } else {
            // segment having a head and a tail to summ up
            shortOutputBuffer[writePosition + k] =
                    longInputBuffer[readHeadPosition + k] + longInputBuffer[readTailPosition + k];
            max = maximum(max, shortOutputBuffer[writePosition + k].real());
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
}