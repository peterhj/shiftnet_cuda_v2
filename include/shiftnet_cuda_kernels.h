/*
Copyright 2017 the shiftnet_cuda_v2 authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include <cuda_runtime.h>

void shiftnet_cuda_moduloshift3x3_nchw_float32(
    float *src,
    float *dst,
    int batch_sz,
    int channels,
    int height,
    int width,
    cudaStream_t stream);

void shiftnet_cuda_moduloshift3x3bwd_nchw_float32(
    float *src,
    float *dst,
    int batch_sz,
    int channels,
    int height,
    int width,
    cudaStream_t stream);

void shiftnet_cuda_moduloshiftgeneric_nchw_float32(
    float *src,
    float *dst,
    int batch_sz,
    int channels,
    int height,
    int width,
    int kernel_size,
    int dilate_factor,
    int direction,
    cudaStream_t stream);
