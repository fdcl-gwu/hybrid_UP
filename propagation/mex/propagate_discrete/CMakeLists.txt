cmake_minimum_required(VERSION 3.20)
project(propagate_discrete LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 14)

include_directories(${PROJECT_SOURCE_DIR}/include)
set(src_dir ${PROJECT_SOURCE_DIR}/src)

find_package(Matlab REQUIRED)
find_package(CUDA REQUIRED)

include_directories("${CUDA_INCLUDE_DIRS}")
include_directories(~/libcutensor/include)

# compile lib
add_library(getLambda STATIC ${src_dir}/setup.cpp ${src_dir}/getLambda.cu)
set_property(TARGET getLambda PROPERTY CUDA_ARCHITECTURES 80)

add_library(getOmega STATIC ${src_dir}/setup.cpp ${src_dir}/getOmega.cu)
set_property(TARGET getOmega PROPERTY CUDA_ARCHITECTURES 80)

add_library(getFcL SHARED ${src_dir}/setup.cpp ${src_dir}/getFcL.cu)
set_property(TARGET getFcL PROPERTY CUDA_ARCHITECTURES 80)

add_library(getIndRule STATIC ${src_dir}/setup.cpp ${src_dir}/getIndRule.cu)
set_property(TARGET getIndRule PROPERTY CUDA_ARCHITECTURES 80)

add_library(propagate STATIC ${src_dir}/setup.cpp ${src_dir}/propagate.cu)
set_property(TARGET propagate PROPERTY CUDA_ARCHITECTURES 80)
target_link_libraries(propagate ~/libcutensor/lib/11/libcutensor.so)

# compile mex
matlab_add_mex(
    NAME getLambda_mex
    SRC ${src_dir}/mexLambda.cpp
    LINK_TO getLambda
    R2018a
)

matlab_add_mex(
    NAME getOmega_mex
    SRC ${src_dir}/mexOmega.cpp
    LINK_TO getOmega
    R2018a
)

matlab_add_mex(
    NAME getFcL_mex
    SRC ${src_dir}/mexFcL.cpp
    LINK_TO getFcL
    R2018a
)

matlab_add_mex(
    NAME getIndRule_mex
    SRC ${src_dir}/mexIndRule.cpp
    LINK_TO getIndRule
    R2018a
)

matlab_add_mex(
    NAME propagate_discrete_mex
    SRC ${src_dir}/mexPropagate.cpp
    LINK_TO propagate
    R2018a
)

