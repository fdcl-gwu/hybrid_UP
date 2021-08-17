#include "getOmega.cuh"

#include "mex.h"

void mexFunction (int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    // get arrays from Matlab
    myReal* R = mymxGetReal(prhs[0]);
    const mwSize* size_R = mxGetDimensions(prhs[0]);

    myReal* x = mymxGetReal(prhs[1]);
    const mwSize* size_x = mxGetDimensions(prhs[1]);
    
    Size_f size_f;
    init_Size_f(&size_f, size_R[2]/2, size_x[1]/2);

    int* indR = mxGetInt32s(prhs[2]);
    const mwSize* size_ind = mxGetDimensions(prhs[2]);
    int numR = size_ind[0];

    myReal* epsilon = mymxGetReal(prhs[3]);
    myReal* PC = mymxGetReal(prhs[4]);
    char* direction = mxArrayToString(prhs[5]);

    // set up output
    size_t size_Omega[4] = {2, (size_t)numR, (size_t)size_f.const_2Bx, (size_t)size_f.const_2Bx};
    plhs[0] = mxCreateUninitNumericArray(4, size_Omega, mymxRealClass, mxREAL);
    myReal* Omega = mymxGetReal(plhs[0]);

    // calculate
    getOmega(Omega, R, x, indR, numR, epsilon, PC, direction, &size_f);

}

