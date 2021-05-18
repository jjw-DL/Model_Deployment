#include <torch/extension.h>

#define CHECK_CUDA(x) do { \
  if (!x.type().is_cuda()) { \
    fprintf(stderr, "%s must be CUDA tensor at %s:%d\n", #x, __FILE__, __LINE__); \
    exit(-1); \
  } \
} while (0)
#define CHECK_CONTIGUOUS(x) do { \
  if (!x.is_contiguous()) { \
    fprintf(stderr, "%s must be contiguous tensor at %s:%d\n", #x, __FILE__, __LINE__); \
    exit(-1); \
  } \
} while (0)
#define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)


// CUDA函数声明
at::Tensor NCReLUForwardLauncher(const at::Tensor& src,
                                 const int batch,
                                 const int channels,
                                 const int height,
                                 const int width);


// C++函数包装
at::Tensor ncrelu_forward_cuda(const at::Tensor input) {
  CHECK_INPUT(input);
  at::DeviceGuard guard(input.device());	
  int batch = input.size(0);
  int channels = input.size(1);
  int height = input.size(2);
  int width = input.size(3);
  return NCReLUForwardLauncher(input, batch, channels, height, width);
}

// 绑定
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ncrelu_forward_cuda", &ncrelu_forward_cuda,"ncrelu forward (CUDA)");
}