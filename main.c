//
//  main.c
//  ParallelTraining
//
//  Created by Hamada M.Zahera on 03/02/2015.
//  Copyright (c) 2015 Hamada M.Zahera. All rights reserved.
//

#ifdef __APPLE__
#include "OpenCL/cl.h"
#else
#include <CL/cl.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, const char * argv[]) {
   
    
    /* Host/Device data structures */
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    
    int errCode;
    
    /* Program/Kernel data structures */
    cl_program program;
    FILE *sourceFile;
    char *program_buffer;
    size_t program_size;
    cl_kernel kernel;
    
    char fileName[]="/Users/hamadazahera/Documents/OpenCLProjects/ParallelTraining/ParallelTraining/parallelTrainingLR.cl";
    char kernalName[]="parallelTrainingLR";
    
    
    
    /* Identify a platform */
    // Requesting one platform.
    errCode=clGetPlatformIDs(1,&platform, NULL);
    if(errCode <0)
    {
        perror("Can't find any platforms");
        exit(1);
    }
    /* Access a device */
    errCode=clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL);
    
    if (errCode<0) {
        perror("Can't access any devices !");
        exit(1);
    }
    
    /* Creating the context */
    context=clCreateContext(NULL, 1, &device, NULL, NULL, &errCode);
    
    if (errCode<0) {
        perror("Can't create any OpenCL devices !");
        exit(1);
    }
    
    /* Reading kernel source from file , loading it on memory
     and build CL program */
    
    sourceFile=fopen(fileName,"r");
    
    if(sourceFile==NULL){
        
        perror("Can't open source file");
        exit(1);
    }
    
    // determine source code size.
    fseek(sourceFile, 0, SEEK_END);
    program_size=ftell(sourceFile);
    
    // Return char pointer back to the begining
    rewind(sourceFile);
    
    // Creating memory buffer and reading file content in it.
    program_buffer=(char *)malloc(program_size+1);
    // Ending file content with a delimiter.
    program_buffer[program_size]='\0';
    
    // Reading file content to program buffer
    fread(program_buffer, sizeof(char), program_size, sourceFile);
    fclose(sourceFile);
    
    // Creating CL program from one source
    program=clCreateProgramWithSource(context,1, (const char **) &program_buffer,&program_size, &errCode);
    
    if (errCode<0) {
        perror("Can't Create CL Program !");
        exit(1);
    }
    
    // Cleaning memory from program buffer.
    free(program_buffer);
    
    
    // Building CL Program
    // If device_list is NULL value, the program executable is build for all devices
    
    clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    
    /* Creating Kernel data structure and I/O buffer */
    kernel=clCreateKernel(program, kernalName, &errCode);
    if (errCode<0) {
        perror("Can't create Kernal !");
        exit(1);
    }
    
    // Loading datasets
    
    long dataSize=100000;
    int numOfAttributes=16;
    long globalSize=dataSize*numOfAttributes;
    
    float samples[globalSize];
    float weights[numOfAttributes];
    float target[dataSize];
   
    // initializing target output values.
    for(int i=0;i<dataSize;i++)
        target[i]=1.0f;
    
    
    float result[dataSize];
    
    // initialize data to be processed by kernals.
    for(int i=0;i<globalSize;i++)
        samples[i]=((double)rand()/RAND_MAX);
    
    for(int i=0;i<numOfAttributes;i++)
        weights[i]=((double)rand()/RAND_MAX); // initial weights with small random values

    // Creating Input and Output buffers in device memory
    cl_mem data_buff, weights_buff,target_buff;
    cl_mem result_buff=NULL;
    
    data_buff=clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, sizeof(float)*globalSize, samples ,  &errCode);
    
    if (errCode<0) {
        perror("Couldn't create a buffer object");
        exit(1);
    }
    
    weights_buff=clCreateBuffer(context, CL_MEM_COPY_HOST_PTR|CL_MEM_READ_ONLY, sizeof(float)*numOfAttributes,weights, NULL);
    
    target_buff=clCreateBuffer(context, CL_MEM_COPY_HOST_PTR|CL_MEM_READ_ONLY ,sizeof(float)*dataSize, target, NULL);
    
    result_buff=clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float)*dataSize, NULL, NULL);
    
    // Sending Arguments to Kernel.
    errCode= clSetKernelArg(kernel, 0, sizeof(cl_mem), &data_buff);
    
    if(errCode<0)
    {
        perror("Couldn't set the kernel argument");
        exit(1);
    }
    
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &target_buff);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &weights_buff);
    
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &result_buff);
    
    /* Creating Command queue. */
    queue=clCreateCommandQueue(context, device, 0, &errCode);
    if (errCode <0) {
        perror("Can't Create Command Queue");
        exit(1);
    }
    
    /* Enqueue kernel and transfer data via command queue. */
    // you have 10 work groups , each one has 16 values as float16.
    
       size_t workItems_per_kernal=dataSize;
    
    cl_event event;

    // Get the current time
    struct timeval foo;
    gettimeofday(&foo,0);
    int startTime=foo.tv_usec;
    printf("Time before kernel in second %ld\n", foo.tv_sec);
    printf("Time before kernel in Microsecond %d\n", foo.tv_usec);

    

    errCode=clEnqueueNDRangeKernel(queue, kernel, 1,NULL,
                                   &workItems_per_kernal, NULL, 0, NULL, &event);
    
    if (errCode<0) {
        perror("Can't enqueue the kernel execution commands");
        exit(1);
    }
    
     /* Reading results from device memory and loading into host.*/
    errCode = clEnqueueReadBuffer(queue, result_buff, CL_TRUE, 0, sizeof(float)*dataSize, result, 0, NULL, NULL);
   
    /* print time after finishing */
    gettimeofday(&foo,NULL);
    printf("Time after kernel in Microsecond %ld\n", foo.tv_sec);
    printf("Time after kernel in Microsecond %d\n", foo.tv_usec);
    
    int diff=foo.tv_usec-startTime;
    printf("Time cost in Microsecond %d\n",diff);

  //handleError("clEnqueueReadBuffer","",status);
    

    if(errCode < 0) {
        perror("Couldn't enqueue the read buffer command");
        exit(1);   
    }
    for (int i=0;i<dataSize; i++) {
        printf("%f",result[i]);
        printf("\n");
    }
    

    /* Deallocating memory resources */
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseProgram(program);
    clReleaseContext(context);
    
   
    

    return 0;
}



