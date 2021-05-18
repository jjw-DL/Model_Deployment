#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <THC/THCAtomics.cuh>

#define THREADS_PER_BLOCK 1024

inline int GET_BLOCKS(const int N) {
  int optimal_block_num = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  int max_block_num = 65000;
  return min(optimal_block_num, max_block_num);
}

template <typename scalar_t>
__global__ void NCReLUForward(const int input_size,
                              const int channels,
                              const int height,
                              const int width,
                              const scalar_t * src_data,
                              scalar_t * dst_data) {

  const int index = blockIdx.x * blockDim.x + threadIdx.x;          // 计算绝对索引

  if (index > input_size) return;
  auto value = src_data[index];                                              // 寻找到原数据值
  const int chw = channels * height * width;
  dst_data[index + index / chw * chw] = value >= 0 ? value : scalar_t(0);             // 前一半通道为正值
  dst_data[index + index / chw * chw + chw] = value >= 0 ? scalar_t(0) : value;    // 后一半通道为负值
} 


at::Tensor NCReLUForwardLauncher(const at::Tensor& src,
                                 const int batch,
                                 const int channels,
                                 const int height,
                                 const int width) {

  at::Tensor dst = at::empty({batch, 2 * channels, height, width},    // 开辟一段存储空间
                            src.options());
  const int input_size = batch * channels * height * width;
  const int output_size = batch * channels * height * width;
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(src.scalar_type(), "NCReLUForwardLauncher", ([&] {
      const scalar_t *src_ = src.data<scalar_t>();
      scalar_t *dst_ = dst.data<scalar_t>();

      NCReLUForward<scalar_t>
        <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK,
        0, at::cuda::getCurrentCUDAStream()>>>(
          input_size, channels, height, width, src_, dst_
        );
  }));
  THCudaCheck(cudaGetLastError());
  return dst;
}