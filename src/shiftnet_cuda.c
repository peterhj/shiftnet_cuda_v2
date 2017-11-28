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

#include "shiftnet_cuda_kernels.h"

#include <THC/THC.h>

#include <stdio.h>

extern THCState *state;

int moduloshift3x3_nchw(THCudaTensor *src_tensor, THCudaTensor *dst_tensor) {
  cudaStream_t stream = THCState_getCurrentStream(state);

  // TODO: support for generic storage types.
  float *src_dptr = THCudaTensor_data(state, src_tensor);
  float *dst_dptr = THCudaTensor_data(state, dst_tensor);

  // Check tensors are 4D.
  int src_ndim = THCudaTensor_nDimension(state, src_tensor);
  int dst_ndim = THCudaTensor_nDimension(state, dst_tensor);
  if (4 != src_ndim) {
    return 0;
  }
  if (4 != dst_ndim) {
    return 0;
  }

  // Check tensor sizes match.
  long src_batch_sz = THCudaTensor_size(state, src_tensor, 0);
  long src_channels = THCudaTensor_size(state, src_tensor, 1);
  long src_height = THCudaTensor_size(state, src_tensor, 2);
  long src_width = THCudaTensor_size(state, src_tensor, 3);
  long dst_batch_sz = THCudaTensor_size(state, dst_tensor, 0);
  long dst_channels = THCudaTensor_size(state, dst_tensor, 1);
  long dst_height = THCudaTensor_size(state, dst_tensor, 2);
  long dst_width = THCudaTensor_size(state, dst_tensor, 3);
  if (src_batch_sz != dst_batch_sz) {
    return 0;
  }
  if (src_channels != dst_channels) {
    return 0;
  }
  if (src_height != dst_height) {
    return 0;
  }
  if (src_width != dst_width) {
    return 0;
  }

  // Check tensor strides are packed.
  long src_batch_sz_stride = THCudaTensor_stride(state, src_tensor, 0);
  long src_channels_stride = THCudaTensor_stride(state, src_tensor, 1);
  long src_height_stride = THCudaTensor_stride(state, src_tensor, 2);
  long src_width_stride = THCudaTensor_stride(state, src_tensor, 3);
  long dst_batch_sz_stride = THCudaTensor_stride(state, dst_tensor, 0);
  long dst_channels_stride = THCudaTensor_stride(state, dst_tensor, 1);
  long dst_height_stride = THCudaTensor_stride(state, dst_tensor, 2);
  long dst_width_stride = THCudaTensor_stride(state, dst_tensor, 3);
  long packed_stride_d0 = 1;
  long packed_stride_d1 = packed_stride_d0 * src_width;
  long packed_stride_d2 = packed_stride_d1 * src_height;
  long packed_stride_d3 = packed_stride_d2 * src_channels;
  if (packed_stride_d0 != src_width_stride || packed_stride_d0 != dst_width_stride) {
    return 0;
  }
  if (packed_stride_d1 != src_height_stride || packed_stride_d1 != dst_height_stride) {
    return 0;
  }
  if (packed_stride_d2 != src_channels_stride || packed_stride_d2 != dst_channels_stride) {
    return 0;
  }
  if (packed_stride_d3 != src_batch_sz_stride || packed_stride_d3 != dst_batch_sz_stride) {
    return 0;
  }

  //printf("DEBUG: moduloshift3x3_nchw: passed size checks\n");
  shiftnet_cuda_moduloshift3x3_nchw_float32(
      src_dptr,
      dst_dptr,
      src_batch_sz,
      src_channels,
      src_height,
      src_width,
      stream);

  return 1;
}

int moduloshift3x3bwd_nchw(THCudaTensor *src_tensor, THCudaTensor *dst_tensor) {
  cudaStream_t stream = THCState_getCurrentStream(state);

  // TODO: support for generic storage types.
  float *src_dptr = THCudaTensor_data(state, src_tensor);
  float *dst_dptr = THCudaTensor_data(state, dst_tensor);

  // Check tensors are 4D.
  int src_ndim = THCudaTensor_nDimension(state, src_tensor);
  int dst_ndim = THCudaTensor_nDimension(state, dst_tensor);
  if (4 != src_ndim) {
    return 0;
  }
  if (4 != dst_ndim) {
    return 0;
  }

  // Check tensor sizes match.
  long src_batch_sz = THCudaTensor_size(state, src_tensor, 0);
  long src_channels = THCudaTensor_size(state, src_tensor, 1);
  long src_height = THCudaTensor_size(state, src_tensor, 2);
  long src_width = THCudaTensor_size(state, src_tensor, 3);
  long dst_batch_sz = THCudaTensor_size(state, dst_tensor, 0);
  long dst_channels = THCudaTensor_size(state, dst_tensor, 1);
  long dst_height = THCudaTensor_size(state, dst_tensor, 2);
  long dst_width = THCudaTensor_size(state, dst_tensor, 3);
  if (src_batch_sz != dst_batch_sz) {
    return 0;
  }
  if (src_channels != dst_channels) {
    return 0;
  }
  if (src_height != dst_height) {
    return 0;
  }
  if (src_width != dst_width) {
    return 0;
  }

  // Check tensor strides are packed.
  long src_batch_sz_stride = THCudaTensor_stride(state, src_tensor, 0);
  long src_channels_stride = THCudaTensor_stride(state, src_tensor, 1);
  long src_height_stride = THCudaTensor_stride(state, src_tensor, 2);
  long src_width_stride = THCudaTensor_stride(state, src_tensor, 3);
  long dst_batch_sz_stride = THCudaTensor_stride(state, dst_tensor, 0);
  long dst_channels_stride = THCudaTensor_stride(state, dst_tensor, 1);
  long dst_height_stride = THCudaTensor_stride(state, dst_tensor, 2);
  long dst_width_stride = THCudaTensor_stride(state, dst_tensor, 3);
  long packed_stride_d0 = 1;
  long packed_stride_d1 = packed_stride_d0 * src_width;
  long packed_stride_d2 = packed_stride_d1 * src_height;
  long packed_stride_d3 = packed_stride_d2 * src_channels;
  if (packed_stride_d0 != src_width_stride || packed_stride_d0 != dst_width_stride) {
    return 0;
  }
  if (packed_stride_d1 != src_height_stride || packed_stride_d1 != dst_height_stride) {
    return 0;
  }
  if (packed_stride_d2 != src_channels_stride || packed_stride_d2 != dst_channels_stride) {
    return 0;
  }
  if (packed_stride_d3 != src_batch_sz_stride || packed_stride_d3 != dst_batch_sz_stride) {
    return 0;
  }

  //printf("DEBUG: moduloshift3x3_nchw: passed size checks\n");
  shiftnet_cuda_moduloshift3x3bwd_nchw_float32(
      src_dptr,
      dst_dptr,
      src_batch_sz,
      src_channels,
      src_height,
      src_width,
      stream);

  return 1;
}

int moduloshiftgeneric_nchw(THCudaTensor *src_tensor, THCudaTensor *dst_tensor, int kernel_size, int dilate_factor, int direction) {
  cudaStream_t stream = THCState_getCurrentStream(state);

  // TODO: support for generic storage types.
  float *src_dptr = THCudaTensor_data(state, src_tensor);
  float *dst_dptr = THCudaTensor_data(state, dst_tensor);

  // Check tensors are 4D.
  int src_ndim = THCudaTensor_nDimension(state, src_tensor);
  int dst_ndim = THCudaTensor_nDimension(state, dst_tensor);
  if (4 != src_ndim) {
    return 0;
  }
  if (4 != dst_ndim) {
    return 0;
  }

  // Check tensor sizes match.
  long src_batch_sz = THCudaTensor_size(state, src_tensor, 0);
  long src_channels = THCudaTensor_size(state, src_tensor, 1);
  long src_height = THCudaTensor_size(state, src_tensor, 2);
  long src_width = THCudaTensor_size(state, src_tensor, 3);
  long dst_batch_sz = THCudaTensor_size(state, dst_tensor, 0);
  long dst_channels = THCudaTensor_size(state, dst_tensor, 1);
  long dst_height = THCudaTensor_size(state, dst_tensor, 2);
  long dst_width = THCudaTensor_size(state, dst_tensor, 3);
  if (src_batch_sz != dst_batch_sz) {
    return 0;
  }
  if (src_channels != dst_channels) {
    return 0;
  }
  if (src_height != dst_height) {
    return 0;
  }
  if (src_width != dst_width) {
    return 0;
  }

  // Check tensor strides are packed.
  long src_batch_sz_stride = THCudaTensor_stride(state, src_tensor, 0);
  long src_channels_stride = THCudaTensor_stride(state, src_tensor, 1);
  long src_height_stride = THCudaTensor_stride(state, src_tensor, 2);
  long src_width_stride = THCudaTensor_stride(state, src_tensor, 3);
  long dst_batch_sz_stride = THCudaTensor_stride(state, dst_tensor, 0);
  long dst_channels_stride = THCudaTensor_stride(state, dst_tensor, 1);
  long dst_height_stride = THCudaTensor_stride(state, dst_tensor, 2);
  long dst_width_stride = THCudaTensor_stride(state, dst_tensor, 3);
  long packed_stride_d0 = 1;
  long packed_stride_d1 = packed_stride_d0 * src_width;
  long packed_stride_d2 = packed_stride_d1 * src_height;
  long packed_stride_d3 = packed_stride_d2 * src_channels;
  if (packed_stride_d0 != src_width_stride || packed_stride_d0 != dst_width_stride) {
    return 0;
  }
  if (packed_stride_d1 != src_height_stride || packed_stride_d1 != dst_height_stride) {
    return 0;
  }
  if (packed_stride_d2 != src_channels_stride || packed_stride_d2 != dst_channels_stride) {
    return 0;
  }
  if (packed_stride_d3 != src_batch_sz_stride || packed_stride_d3 != dst_batch_sz_stride) {
    return 0;
  }

  if (kernel_size <= 0) {
    return 0;
  }
  if (dilate_factor <= 0) {
    return 0;
  }
  if (direction != 1 && direction != -1) {
    return 0;
  }

  int dilated_half_kernel_size = dilate_factor * (kernel_size / 2);
  int tile_out_size = 16 - 2 * dilated_half_kernel_size;
  if (tile_out_size <= 0) {
    return 0;
  }

  //printf("DEBUG: moduloshiftgeneric_nchw: passed size checks: %d %d %d\n",
  //    kernel_size, dilate_factor, direction);
  shiftnet_cuda_moduloshiftgeneric_nchw_float32(
      src_dptr,
      dst_dptr,
      src_batch_sz,
      src_channels,
      src_height,
      src_width,
      kernel_size,
      dilate_factor,
      direction,
      stream);

  return 1;
}
