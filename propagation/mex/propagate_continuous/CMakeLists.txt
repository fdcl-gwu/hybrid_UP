cmake_minimum_required(VERSION 3.15)
project(propagate_continuous LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
find_package(Matlab)
find_package(CUDA REQUIRED)
set(CMAKE_CUDA_STANDARD 14)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")

set(CMAKE_CXX_STANDARD 11)

matlab_add_mex(
    NAME propagate_continuous_mex
    SRC kernel.cu setup.cu integrate.cu fftSO3R.cu
    LINK_TO cufft cublas
    R2018a
)

set_property(TARGET propagate_continuous_mex PROPERTY CUDA_ARCHITECTURES 80)

include_directories(~/libcutensor/include)

target_link_libraries(propagate_continuous_mex ~/libcutensor/lib/11/libcutensor.so)

