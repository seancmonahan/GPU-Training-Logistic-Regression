#ifdef FP_64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif
/* if USE_DOUBLE16 is defined, the kernel uses vector operations which
 hopefully are faster than scalar operations. However, this does not
 seem to give any advantage on CPU based OpenCL implementations */
#undef USE_DOUBLE16

__kernel void parallelTrainingLR( __global float16* A, __global float* t, __global float16* weight, __global float16* deltaWeight) {

     int i=get_global_id(0);
      int thread = get_global_id(0);
    
    // create local sum
    float sum=0.0f;
    
    float sum1=dot(A[i].lo.lo,weight[i].lo.lo);
    float sum2=dot(A[i].lo.hi,weight[i].lo.hi);
    float sum3=dot(A[i].hi.lo,weight[i].hi.lo);
    float sum4=dot(A[i].hi.hi,weight[i].hi.hi);
    
    sum=sum1+sum2+sum3+sum4;
    
    // estimated error
    float e=0.0f;
    float out=0.0f;
     // actual output out
    out=1.0f/(1.0f+exp(-sum));
    
    
    e=t[i]-out;
    barrier( CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE );
    if(thread==0)
        // learning rate : 0.7 (fixed value)
        deltaWeight[i]=e*A[i]*0.7f;
    
}