## 编写一个 C++/CUDA 混合的拓展

编写 CUDA 扩展的一般策略是首先编写一个 C++ 文件，该文件定义了将从 Python 中调用的函数，并使用 pybind11 将这些函数绑定到 Python 上。此外，该文件还将 *声明* 在 CUDA(`.cu`）文件中将定义的函数。然后，C++ 函数将进行一些检查，并最终将其调用转发给 CUDA 函数。在 CUDA 文件中，我们编写了实际的 CUDA 内核。接着，`cpp_extension`包将负责使用 C++ 编译器(如`gcc`）和使用 NVIDIA 的`nvcc`编译器的CUDA源编译 C++ 源代码。以此来确保每个编译器处理它最好编译的文件。最终，它们将链接到一个可从 Python 代码中进行访问的共享库。

### 编写setup.py文件

```python
from setuptools import setup
from torch.utils import cpp_extension

setup(
    name='ncrelu_cpp',  # 编译后的链接库名称
    ext_modules=[cpp_extension.CppExtension(
        'ncrelu_cpp',['ncrelu.cpp']  # 待编译文件，及编译函数
    )],
    cmdclass={  # 执行编译命令设置
        'build_ext':cpp_extension.BuildExtension
    }
)
```

我们现在使用`CUDAExtension()`而不是`CppExtension()`。我们可以只指定`.cu`文件和`.cpp`文件——库可以解决所有麻烦。JIT 机制则更简单：

```python
from torch.utils.cpp_extension import load

lltm = load(name='ncrelu_cuda', sources=['ncrelu_cuda.cpp', 'ncrelu_cuda_kernel.cu'])
```

### 编写c++文件(ncrelu_cuda.cpp)

该文件将声明在 CUDA(`.cu`）文件中将定义的函数。然后，C++ 函数将进行一些检查，并最终将其调用转发给 CUDA 函数，最终使用 pybind11 将这些函数绑定到Python上。

```c++
#include <torch/extension.h>

// CUDA函数声明
at::Tensor NCReLUForwardLauncher(const at::Tensor& src,
                                 const int batch,
                                 const int channels,
                                 const int height,
                                 const int width);

// 宏定义
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

// C++函数包装,最后编译出的动态链接库可调用的函数名
at::Tensor ncrelu_forward_cuda(const at::Tensor input) {
  CHECK_INPUT(input);
  at::DeviceGuard guard(input.device());	
  int batch = input.size(0);
  int channels = input.size(1);
  int height = input.size(2);
  int width = input.size(3);
  return NCReLUForwardLauncher(input, batch, channels, height, width);
}

// 绑定,换个函数名
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ncrelu_forward_cuda", &ncrelu_forward_cuda,
        "ncrelu forward (CUDA)");
}
```

### 编写cuda文件（ncrelu_cuda_kernel.cu）

整体结构：

```c++
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <THC/THCAtomics.cuh>

template <typename scalar_t>
__global__ void NCReLUForward(/*参数*/) {  // cuda核函数
  /* 具体实现部分 */
}

at::Tensor NCReLUForwardLauncher(/*参数*/) {  // 包装（启动函数）函数
  /* 输入转换，初始化 */
  /* 传入函数中计算，返回结果*/
}
```



```c++
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
```

- 首先函数的输入参数为ATen Tensor类型的源数据，以及数据的四个维度。
- 然后先去开辟一段存储空间，负责承接输出的计算结果。这里也是用到了at::empty高层接口操作，非常简单。然后是计算输入大小
- `AT_DISPATCH_FLOATING_TYPES` 的目的是为我们处理此次调度。它需要**一个类型**(在我们的例子中是`src.scalar_type()`），**一个名称**(用于错误消息）和**一个 lambda 函数**。在这个 lambda 函数中，类型别名`scalar_t`是可用的，并且被定义为张量在该上下文中实际处于运行时的类型。因此，如果我们有一个模板函数(我们的 CUDA 内核将作为模板函数），我们可以用这个`scalar_t`别名实例化它，然后正确的函数就会被调用。 
- 虽然 ATen 会对我们所处理的张量的设备和数据类型进行抽象化，但是在运行时，张量仍将由具体设备上的具体类型的内存支持。因此，我们需要一种在运行时确定张量是什么类型的方法，然后选择性地调用相应的具有正确类型签名(signature）函数。手动完成这些部分，这将(概念上）看起来像这样：

```c++
switch (tensor.type().scalarType()) {
  case at::ScalarType::Double:
    return function<double>(tensor.data<double>());
  case at::ScalarType::Float:
    return function<float>(tensor.data<float>());
  ...
}
```

- 在从ATen Tensor中获取某一类型数据指针后，用到了`<<< >>>`这一写法启动kernel。其中需要根据输出大小分配block数，并设置每一block中的thread数
- 最后再调用了CUDA kernel计算结果之后，进行最后的检查，如果无报错则返回结果。
