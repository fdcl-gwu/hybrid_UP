## Uncertainty Propagation for General Stochastic Hybrid System (GSHS)

### What does this repository do?
A GSHS is a stochastic dynamical system that has both continuous and discrete dynamics.
This repository provides code for uncertainty propagation of a specific example of GSHS: a 3D pendulum that collides with a planar wall.

### How to use the code?
The code consist of two parts: (i) a Matlab interface that sets parameters, saves and plots data. (ii) C code that use cuda to perform actual computations. The C code are built into binary mex files that can be called directly in Matlab.

There are four main functions in the `propagation` folder:
```
pendulum_continuous.m
pendulum_hybrid.m
pendulum_continuous_MC.m
pendulum_hybrid_MC.m
```
The first two functions use the proposed splitting-spectral method, and the last two use Monte Carlo simulations.
For the two `continuous` functions, only the continuous dynamics is propagated, i.e., the pendulum does not collide with the wall.
For the two `hybrid` functions, both the continuous and discrete dynamics are propagated.
The function `pendulum_continuous` can be either entirely in Matlab, or with those intensive computations in cuda, depending on the parameter `use_mex`.
The function `pendulum_hybrid` can only perform computations in cuda.
The last two `MC` functions are entirely in Matlab.

The functions entirely in Matlab can be run anywhere Matlab is installed, together with a cuda enabled GPU (using Matlab built-in GPU supports).
The functions that depends on the cuda code in this repository can only be run in Linux.

### Parameter lists
```
[stat, MFG] = pendulum_continuous(path, use_mex, method, getc, f0)
```
* `path`: the folder used to store density values.
          If empty, the density values are discarded.
* `use_mex`: If `true`, use the mex files with computations in cuda, defaults to false.
* `method`: a string deciding the numerical integration method used to propagate continuous dynamics:
  * 'euler': forward Euler's method.
  * 'midpoint': mid-point Euler's method.
  * 'RK2': second order Runge-Kutta method.
  * 'RK4': fourth order Runge-Kutta method.
* `getc`: If `true`, the density values are converted to a set of colors for visualization, and then saved to disk.
          The density values are only saved to disk every four iterations to save disk space.
          Defaults to false.
* `f0`: Initial density values.
        If empty, default initial density values are used.
* `stat`: Descriptive statistics of the density values.
* `MFG`: Match the density values to matrix Fisher-Gaussian distributions.

```
[stat, MFG, R_res, x_res] = pendulum_continuous_MC(R, x)
```
* `R`, `x`: Initial samples of attitude and angular velocity.
            If empty, initial samples are generated from default initial density values.
* `R_res`, `x_res`: Sample trajectories of 1000 samples.

### How to visualize density values
All function for visualization are in the folder `propagation/plotting/`.
If `getc == true` when computing density values, use the `plot_continuous` function for continuous dynamics, and `plot_hybrid` for hybrid dynamics, for visualization of marginal attitude densities.
If `getc == false` when computing density values, use the `pendulum_plot` function for visualization of marginal attitude densities.
Use the `plot_Omega` function for visualization of marginal angular velocity densities.

### Dependecies
* For Matlab only functions, Matlab R2021a and parallel computing toolbox is required.
* To use mex files for cuda computation, the followings are needed:
  * Nvidia GPU Computing Toolkit 11.2
  * `cuTensor 1.2.2`: place the library to `~/libcutensor/`
* To complie the mex file for `getColor`, used when `getc == true`, the followings libraries are needed:
  * OpenMP
  * fftw

### Hardware limitations
The parameters `BR` and `Bx` are the band-limits for the Fourier transform on SO(3) and R^2, they have strong impacts on the computational resourses needed.
When `BR = Bx <= 15`, the GPU onboard memory required is less than 6GB, so the code can be run on a typical gaming GPU.
When `BR = Bx = 30` (the largest value we have tested), the GPU onboard memory required is less than 40GB, so the code can be run on a single Nvidia A100-40GB GPU.
Also note that the computations are all in double precision, so it will be very beneficial if the GPU is dedicated to FP64, especially if the GPU has FP64 tensor cores.

### Build binaries from source C code
Use the `CMakeLists.txt` in each source code folder, and standard CMake instructions:
```
mkdir build
cd build
cmake ..
make
```
The binaries in this repository are built using CMake 3.20.0.
